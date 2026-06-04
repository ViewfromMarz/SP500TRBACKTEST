from datetime import date
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="GSPC Backtest Lab", layout="wide")

st.title("GSPC backtest lab")
st.caption(
    "Backtest lab for ^GSPC strategies using FRED cash rates and optional FRED real-rate regime filters. "
    "Constant leverage mode supports tiered leverage rules based on the published FRED 1-year or 10-year real-rate series. "
    "A research mode also compares S&P 500 dividend yield against key rates."
)


@st.cache_data(show_spinner=False)
def load_fred_series(series_id, start, end):
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start.strftime('%Y-%m-%d')}&coed={end.strftime('%Y-%m-%d')}"
    )
    df = pd.read_csv(url)
    df.columns = ["Date", "Value"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df.set_index("Date").sort_index()


@st.cache_data(show_spinner=False)
def load_monthly_fred_series(series_id, start, end, output_col):
    df = load_fred_series(series_id, start, end).rename(columns={"Value": output_col}).copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    monthly = df.resample("MS").last()
    monthly[output_col] = pd.to_numeric(monthly[output_col], errors="coerce")
    return monthly


@st.cache_data(show_spinner=False)
def load_uploaded_dividend_yield_data(file_bytes):
    raw = pd.read_excel(io.BytesIO(file_bytes))
    raw.columns = [str(c).strip() for c in raw.columns]
    lower_map = {c.lower(): c for c in raw.columns}

    date_col = None
    for candidate in ["date", "month"]:
        if candidate in lower_map:
            date_col = lower_map[candidate]
            break
    if date_col is None:
        possible = [c for c in raw.columns if "date" in c.lower() or "month" in c.lower()]
        if possible:
            date_col = possible[0]

    div_col = None
    for candidate in ["div yield", "dividend yield", "divident yiled", "sp500 dividend yield"]:
        if candidate in lower_map:
            div_col = lower_map[candidate]
            break
    if div_col is None:
        possible = [c for c in raw.columns if "div" in c.lower() and "yield" in c.lower()]
        if possible:
            div_col = possible[0]

    if date_col is None or div_col is None:
        raise ValueError(f"Could not identify dividend-yield file columns. Found columns: {list(raw.columns)}")

    df = raw[[date_col, div_col]].copy()
    df.columns = ["date_raw", "sp500_dividend_yield"]

    date_text = df["date_raw"].astype(str).str.strip()
    date_text = date_text.str.replace(r"[^0-9.]", "", regex=True)
    date_text = date_text.str.replace(r"^(\d{4})\.(\d)$", r"\1.0\2", regex=True)
    parsed = pd.to_datetime(date_text, format="%Y.%m", errors="coerce")
    fallback = pd.to_datetime(df["date_raw"], errors="coerce")

    df["date"] = parsed.fillna(fallback)
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    df["sp500_dividend_yield"] = pd.to_numeric(df["sp500_dividend_yield"], errors="coerce")
    df = df.dropna(subset=["date", "sp500_dividend_yield"]).copy()
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df[["date", "sp500_dividend_yield"]]


def build_dividend_delta_signal(start, end, uploaded_file_bytes):
    if uploaded_file_bytes is None:
        raise ValueError("Please upload a dividend-yield file to use the Dividend Yield Delta signal.")

    pad_start = pd.to_datetime(start) - pd.DateOffset(years=2)
    pad_end = pd.to_datetime(end) + pd.DateOffset(months=1)

    div_df = load_uploaded_dividend_yield_data(uploaded_file_bytes)
    div_df = div_df.loc[(div_df["date"] >= pad_start.to_period("M").to_timestamp()) & (div_df["date"] <= pad_end.to_period("M").to_timestamp())].copy()

    gs1 = load_monthly_fred_series("GS1", pad_start, pad_end, "gs1")
    gs10 = load_monthly_fred_series("GS10", pad_start, pad_end, "gs10")
    real_1y = load_monthly_fred_series("REAINTRATREARAT1YE", pad_start, pad_end, "real_1y")
    real_10y = load_monthly_fred_series("REAINTRATREARAT10Y", pad_start, pad_end, "real_10y")

    rates_df = gs1.join(gs10, how="outer").join(real_1y, how="outer").join(real_10y, how="outer")
    rates_df = rates_df.reset_index().rename(columns={"Date": "date"})
    rates_df["date"] = pd.to_datetime(rates_df["date"]).dt.to_period("M").dt.to_timestamp()

    merged = pd.merge(div_df, rates_df, on="date", how="left").sort_values("date").reset_index(drop=True)
    merged["div_minus_gs1"] = merged["sp500_dividend_yield"] - merged["gs1"]
    merged["div_minus_gs10"] = merged["sp500_dividend_yield"] - merged["gs10"]
    merged["div_minus_real_1y"] = merged["sp500_dividend_yield"] - merged["real_1y"]
    merged["div_minus_real_10y"] = merged["sp500_dividend_yield"] - merged["real_10y"]
    return merged


def map_dividend_delta_to_daily(df, monthly_signal_df, signal_col):
    out = df.copy()
    monthly = monthly_signal_df[["date", signal_col]].copy().rename(columns={signal_col: "Dividend_Delta_Signal"})
    monthly = monthly.sort_values("date")
    out = out.reset_index().rename(columns={"index": "Date"})
    out["signal_month"] = pd.to_datetime(out["Date"]).dt.to_period("M").dt.to_timestamp()
    out = out.merge(monthly, left_on="signal_month", right_on="date", how="left")
    out = out.drop(columns=["signal_month", "date"])
    out = out.set_index("Date")
    out["Dividend_Delta_Signal"] = pd.to_numeric(out["Dividend_Delta_Signal"], errors="coerce").ffill() / 100.0
    return out


def select_trigger_value(row, trigger_mode, real_rate_source):
    if trigger_mode == "Dividend Yield Delta":
        return float(row["Dividend_Delta_Signal"])
    return select_real_rate(row, real_rate_source)


def determine_tiered_allocation_from_trigger(trigger_value, tier1_threshold_pct, tier1_alloc_pct, tier2_threshold_pct, tier2_alloc_pct, base_alloc_pct, trigger_mode):
    base_alloc = base_alloc_pct / 100.0
    tier1_alloc = tier1_alloc_pct / 100.0
    tier2_alloc = tier2_alloc_pct / 100.0

    if pd.isna(trigger_value):
        return base_alloc, "No trigger data"

    if trigger_mode == "Dividend Yield Delta":
        if trigger_value > (tier2_threshold_pct / 100.0):
            return tier2_alloc, f"Div delta > {tier2_threshold_pct:.2f}%"
        if trigger_value > (tier1_threshold_pct / 100.0):
            return tier1_alloc, f"Div delta > {tier1_threshold_pct:.2f}%"
        return base_alloc, "Base allocation"

    if trigger_value < (tier2_threshold_pct / 100.0):
        return tier2_alloc, f"Real rate < {tier2_threshold_pct:.2f}%"
    if trigger_value < (tier1_threshold_pct / 100.0):
        return tier1_alloc, f"Real rate < {tier1_threshold_pct:.2f}%"
    return base_alloc, "Base allocation"


@st.cache_data(show_spinner=False)
def load_dividend_yield_rates_data(start, end):
    pad_start = pd.to_datetime(start) - pd.DateOffset(years=2)
    pad_end = pd.to_datetime(end) + pd.DateOffset(months=1)

    div_df = load_shiller_dividend_yield_data(pad_start, pad_end)
    fedfunds = load_monthly_fred_series("FEDFUNDS", pad_start, pad_end, "fedfunds")
    gs1 = load_monthly_fred_series("GS1", pad_start, pad_end, "gs1")
    gs10 = load_monthly_fred_series("GS10", pad_start, pad_end, "gs10")
    real_1y = load_monthly_fred_series("REAINTRATREARAT1YE", pad_start, pad_end, "real_1y")
    real_10y = load_monthly_fred_series("REAINTRATREARAT10Y", pad_start, pad_end, "real_10y")

    rates_df = fedfunds.join(gs1, how="outer")
    rates_df = rates_df.join(gs10, how="outer")
    rates_df = rates_df.join(real_1y, how="outer")
    rates_df = rates_df.join(real_10y, how="outer")
    rates_df = rates_df.reset_index().rename(columns={"Date": "date"})
    rates_df["date"] = pd.to_datetime(rates_df["date"]).dt.to_period("M").dt.to_timestamp()

    merged = pd.merge(div_df, rates_df, on="date", how="outer")
    merged = merged[
        [
            "date",
            "sp500_dividend_yield",
            "fedfunds",
            "gs1",
            "gs10",
            "real_1y",
            "real_10y",
        ]
    ].sort_values("date").reset_index(drop=True)
    return merged


@st.cache_data(show_spinner=False)
def load_shiller_dividend_yield_data(start, end):
    # DataHub exposes a cleaned monthly CSV derived from Robert Shiller's dataset.
    # The column naming can occasionally vary, so the parser below is defensive.
    url = "https://datahub.io/core/s-and-p-500/r/data.csv"
    df = pd.read_csv(url)
    df.columns = [str(c).strip() for c in df.columns]

    lower_map = {c.lower(): c for c in df.columns}

    date_col = None
    for candidate in ["date", "year"]:
        if candidate in lower_map:
            date_col = lower_map[candidate]
            break

    if date_col is None:
        possible_date_cols = [c for c in df.columns if "date" in c.lower() or "year" in c.lower()]
        if possible_date_cols:
            date_col = possible_date_cols[0]

    dividend_col = None
    price_col = None
    explicit_yield_col = None

    for c in df.columns:
        cl = c.lower()
        if cl in ["dividend_yield", "div yield", "yield", "dy"]:
            explicit_yield_col = c
        if cl in ["dividend", "dividends", "d"]:
            dividend_col = c
        if cl in ["price", "p", "sp500", "s&p 500"]:
            price_col = c

    if date_col is None:
        raise ValueError("Could not identify a date column in the built-in Shiller/DataHub dataset.")

    shiller = df.copy()
    shiller[date_col] = pd.to_numeric(shiller[date_col], errors="coerce")
    shiller = shiller.dropna(subset=[date_col]).copy()

    def _parse_decimal_year(val):
        year = int(val)
        frac = float(val) - year
        month = int(round(frac * 12))
        month = min(max(month, 1), 12)
        return pd.Timestamp(year=year, month=month, day=1)

    shiller["date"] = shiller[date_col].apply(_parse_decimal_year)

    if explicit_yield_col is not None:
        shiller["sp500_dividend_yield"] = pd.to_numeric(shiller[explicit_yield_col], errors="coerce")
    elif dividend_col is not None and price_col is not None:
        shiller[dividend_col] = pd.to_numeric(shiller[dividend_col], errors="coerce")
        shiller[price_col] = pd.to_numeric(shiller[price_col], errors="coerce")
        shiller["sp500_dividend_yield"] = np.where(
            shiller[price_col] > 0,
            (shiller[dividend_col] / shiller[price_col]) * 100.0,
            np.nan,
        )
    else:
        raise ValueError(
            "Could not identify dividend yield, or dividend/price columns, in the built-in Shiller/DataHub dataset."
        )

    shiller["date"] = pd.to_datetime(shiller["date"]).dt.to_period("M").dt.to_timestamp()
    shiller = shiller[["date", "sp500_dividend_yield"]].copy()
    shiller = shiller.dropna(subset=["date", "sp500_dividend_yield"])
    shiller = shiller.sort_values("date").reset_index(drop=True)
    shiller = shiller.loc[
        (shiller["date"] >= pd.to_datetime(start).to_period("M").to_timestamp())
        & (shiller["date"] <= pd.to_datetime(end).to_period("M").to_timestamp())
    ].copy()
    return shiller



def make_dividend_yield_rates_chart(df, selected_series):
    fig = go.Figure()
    for col in selected_series:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[col],
                mode="lines",
                name=col,
            )
        )

    fig.update_layout(
        height=750,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis_title="Date",
        yaxis_title="Percent",
    )
    return fig



def normalize_download(df, prefix):
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    keep = [c for c in ["Open", "Close", "Adj Close"] if c in df.columns]
    df = df[keep].copy()
    return df.rename(
        columns={
            "Open": f"{prefix}_Open",
            "Close": f"{prefix}_Close",
            "Adj Close": f"{prefix}_Adj_Close",
        }
    )


@st.cache_data(show_spinner=False)
def load_gspc_data(start, end):
    pad_start = pd.to_datetime(start) - pd.Timedelta(days=400)
    pad_end = pd.to_datetime(end) + pd.Timedelta(days=7)

    gspc_raw = yf.download(
        "^GSPC",
        start=pad_start.strftime("%Y-%m-%d"),
        end=(pad_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    sp500tr_raw = yf.download(
        "^SP500TR",
        start=pad_start.strftime("%Y-%m-%d"),
        end=(pad_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if gspc_raw.empty:
        raise ValueError("Unable to download ^GSPC data.")

    tb3ms = load_fred_series("TB3MS", pad_start, pad_end).rename(columns={"Value": "Cash_Yield"})
    real_1y = load_fred_series("REAINTRATREARAT1YE", pad_start, pad_end).rename(columns={"Value": "Real_1Y_FRED"})
    real_10y = load_fred_series("REAINTRATREARAT10Y", pad_start, pad_end).rename(columns={"Value": "Real_10Y_FRED"})

    gspc = normalize_download(gspc_raw, "GSPC")
    gspc = gspc.dropna(subset=["GSPC_Open", "GSPC_Close", "GSPC_Adj_Close"]).copy()

    if sp500tr_raw.empty:
        sp500tr = pd.DataFrame(index=gspc.index, data={"SP500TR_Close": np.nan})
    else:
        sp500tr = normalize_download(sp500tr_raw, "SP500TR")
        if "SP500TR_Close" not in sp500tr.columns:
            sp500tr = pd.DataFrame(index=gspc.index, data={"SP500TR_Close": np.nan})
        else:
            sp500tr = sp500tr[["SP500TR_Close"]].copy()

    df = gspc.join(sp500tr, how="left")
    df = df.join(tb3ms[["Cash_Yield"]], how="left")
    df = df.join(real_1y[["Real_1Y_FRED"]], how="left")
    df = df.join(real_10y[["Real_10Y_FRED"]], how="left")

    df["Cash_Yield"] = pd.to_numeric(df["Cash_Yield"], errors="coerce").ffill() / 100.0
    df["Real_1Y_FRED"] = pd.to_numeric(df["Real_1Y_FRED"], errors="coerce").ffill() / 100.0
    df["Real_10Y_FRED"] = pd.to_numeric(df["Real_10Y_FRED"], errors="coerce").ffill() / 100.0
    df["SP500TR_Close"] = pd.to_numeric(df["SP500TR_Close"], errors="coerce").ffill()

    df = df.dropna(
        subset=["GSPC_Open", "GSPC_Close", "GSPC_Adj_Close", "Cash_Yield", "Real_1Y_FRED", "Real_10Y_FRED"]
    ).copy()

    df["GSPC_Ret"] = df["GSPC_Close"].pct_change().fillna(0.0)
    df["SP500TR_Ret"] = df["SP500TR_Close"].pct_change().fillna(0.0)
    df.loc[df["SP500TR_Close"].isna(), "SP500TR_Ret"] = np.nan
    df["Cash_Daily_Ret"] = (1.0 + df["Cash_Yield"]).pow(1.0 / 252.0) - 1.0
    return df



def choose_return_stream(df, reinvest_dividends):
    out = df.copy()
    use_tr = reinvest_dividends and out["SP500TR_Ret"].notna().any()
    out["Asset_Ret"] = out["SP500TR_Ret"] if use_tr else out["GSPC_Ret"]
    out["Benchmark_Ret"] = out["Asset_Ret"]
    out["Benchmark_Label"] = "Buy & Hold SP500TR" if use_tr else "Buy & Hold GSPC"
    return out, use_tr



def target_alloc(v, t1, t2, a1, a2, a3):
    if pd.isna(v):
        return a1
    if v < t1:
        return a1
    if v < t2:
        return a2
    return a3



def mark_weekly_signals(index):
    week_period = pd.Series(index.to_period("W-FRI"), index=index)
    last_days = week_period.groupby(week_period).apply(lambda x: x.index.max())
    return pd.Index(last_days.values)



def calc_strategy_return(asset_ret, cash_ret, current_alloc, borrow_spread_bps):
    borrow_rate_daily = cash_ret + (borrow_spread_bps / 10000.0) / 252.0
    if current_alloc <= 1.0:
        strat_ret = current_alloc * asset_ret + (1.0 - current_alloc) * cash_ret
    else:
        borrowed = current_alloc - 1.0
        strat_ret = current_alloc * asset_ret - borrowed * borrow_rate_daily
    return strat_ret, borrow_rate_daily



def select_real_rate(row, real_rate_source):
    if real_rate_source == "1-Year Real Rate (FRED REAINTRATREARAT1YE)":
        return float(row["Real_1Y_FRED"])
    return float(row["Real_10Y_FRED"])



def determine_tiered_allocation(
    real_rate,
    tier1_threshold_pct,
    tier1_alloc_pct,
    tier2_threshold_pct,
    tier2_alloc_pct,
    base_alloc_pct,
):
    base_alloc = base_alloc_pct / 100.0
    tier1_alloc = tier1_alloc_pct / 100.0
    tier2_alloc = tier2_alloc_pct / 100.0

    if pd.isna(real_rate):
        return base_alloc, "No real-rate data"
    if real_rate < (tier2_threshold_pct / 100.0):
        return tier2_alloc, f"Real rate < {tier2_threshold_pct:.2f}%"
    if real_rate < (tier1_threshold_pct / 100.0):
        return tier1_alloc, f"Real rate < {tier1_threshold_pct:.2f}%"
    return base_alloc, "Base allocation"



def run_constant_leverage_backtest(
    df,
    start,
    end,
    initial_capital,
    borrow_spread_bps,
    use_nominal_gate,
    nominal_cap_pct,
    trigger_mode,
    real_rate_source,
    base_alloc_pct,
    tier1_threshold_pct,
    tier1_alloc_pct,
    tier2_threshold_pct,
    tier2_alloc_pct,
):
    data = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()
    if len(data) < 3:
        raise ValueError("Not enough data in selected window.")

    first_trigger = select_trigger_value(data.iloc[0], trigger_mode, real_rate_source)
    first_target_alloc, first_rule = determine_tiered_allocation_from_trigger(
        first_trigger,
        tier1_threshold_pct,
        tier1_alloc_pct,
        tier2_threshold_pct,
        tier2_alloc_pct,
        base_alloc_pct,
        trigger_mode,
    )
    first_nominal = float(data.iloc[0]["Cash_Yield"] + (borrow_spread_bps / 10000.0))
    first_nominal_ok = True if not use_nominal_gate else first_nominal < (nominal_cap_pct / 100.0)
    current_alloc = first_target_alloc if first_nominal_ok else min(first_target_alloc, 1.0)

    strategy_vals = [initial_capital]
    bench_vals = [initial_capital]
    alloc_hist = [current_alloc]
    desired_alloc_hist = [first_target_alloc]
    rule_hist = [first_rule]
    leverage_flag = [current_alloc > 1.0]
    nominal_gate_pass_hist = [first_nominal_ok]
    nominal_borrow_rate_hist = [first_nominal]
    real_rate_hist = [first_trigger]
    borrow_rate_daily_hist = [np.nan]

    idx = data.index.tolist()
    for i in range(1, len(idx)):
        row = data.loc[idx[i]]

        trigger_value = select_trigger_value(row, trigger_mode, real_rate_source)
        target_alloc, rule_label = determine_tiered_allocation_from_trigger(
            trigger_value,
            tier1_threshold_pct,
            tier1_alloc_pct,
            tier2_threshold_pct,
            tier2_alloc_pct,
            base_alloc_pct,
            trigger_mode,
        )
        nominal_borrow_rate = float(row["Cash_Yield"] + (borrow_spread_bps / 10000.0))
        nominal_ok = True if not use_nominal_gate else nominal_borrow_rate < (nominal_cap_pct / 100.0)
        current_alloc = target_alloc if nominal_ok else min(target_alloc, 1.0)

        asset_ret = float(row["Asset_Ret"])
        cash_ret = float(row["Cash_Daily_Ret"])
        strat_ret, borrow_rate_daily = calc_strategy_return(asset_ret, cash_ret, current_alloc, borrow_spread_bps)
        bench_ret = float(row["Benchmark_Ret"])

        strategy_vals.append(strategy_vals[-1] * (1 + strat_ret))
        bench_vals.append(bench_vals[-1] * (1 + bench_ret))
        alloc_hist.append(current_alloc)
        desired_alloc_hist.append(target_alloc)
        rule_hist.append(rule_label)
        leverage_flag.append(current_alloc > 1.0)
        nominal_gate_pass_hist.append(nominal_ok)
        nominal_borrow_rate_hist.append(nominal_borrow_rate)
        real_rate_hist.append(trigger_value)
        borrow_rate_daily_hist.append(borrow_rate_daily)

    out = data.iloc[: len(strategy_vals)].copy()
    out["Strategy_Value"] = strategy_vals
    out["Benchmark_Value"] = bench_vals
    out["Allocation_Pct"] = np.array(alloc_hist) * 100.0
    out["Desired_Allocation_Pct"] = np.array(desired_alloc_hist) * 100.0
    out["Rule_Label"] = rule_hist[: len(out)]
    out["Leverage_Flag"] = leverage_flag[: len(out)]
    out["Nominal_Gate_Pass"] = nominal_gate_pass_hist[: len(out)]
    out["Nominal_Borrow_Rate_Annual"] = nominal_borrow_rate_hist[: len(out)]
    out["Selected_Real_Rate"] = real_rate_hist[: len(out)]
    out["Borrow_Rate_Daily"] = borrow_rate_daily_hist[: len(out)]
    out["Strategy_Return"] = out["Strategy_Value"].pct_change().fillna(0.0)
    out["Benchmark_Return"] = out["Benchmark_Value"].pct_change().fillna(0.0)
    return out



def run_buy_the_dip_backtest(df, start, end, signal_mode, t1, t2, a1, a2, a3, initial_capital, borrow_spread_bps):
    data = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))].copy()
    if len(data) < 3:
        raise ValueError("Not enough data in selected window.")

    data["Running_High_Close"] = data["GSPC_Close"].cummax()
    data["Off_High_Pct"] = (1.0 - data["GSPC_Close"] / data["Running_High_Close"]) * 100.0
    weekly_days = set(mark_weekly_signals(data.index)) if signal_mode == "Weekly" else set(data.index)

    pending_alloc = None
    current_alloc = a1 / 100.0
    strategy_vals = [initial_capital]
    bench_vals = [initial_capital]
    alloc_hist = [current_alloc]
    off_high_hist = [np.nan]
    trade_flag = [False]
    leverage_flag = [current_alloc > 1.0]
    borrow_rate_daily_hist = [np.nan]
    borrow_rate_annual_hist = [np.nan]

    idx = data.index.tolist()
    for i in range(1, len(idx)):
        d = idx[i]
        off_high = float(data.loc[d, "Off_High_Pct"]) if not pd.isna(data.loc[d, "Off_High_Pct"]) else np.nan
        off_high_hist.append(off_high)

        if d in weekly_days:
            pending_alloc = target_alloc(off_high, t1, t2, a1, a2, a3) / 100.0

        traded = False
        if pending_alloc is not None:
            if abs(current_alloc - pending_alloc) > 1e-12:
                traded = True
            current_alloc = pending_alloc

        asset_ret = float(data.loc[d, "Asset_Ret"])
        cash_ret = float(data.loc[d, "Cash_Daily_Ret"])
        strat_ret, borrow_rate_daily = calc_strategy_return(asset_ret, cash_ret, current_alloc, borrow_spread_bps)
        bench_ret = float(data.loc[d, "Benchmark_Ret"])

        strategy_vals.append(strategy_vals[-1] * (1 + strat_ret))
        bench_vals.append(bench_vals[-1] * (1 + bench_ret))
        alloc_hist.append(current_alloc)
        trade_flag.append(traded)
        leverage_flag.append(current_alloc > 1.0)
        borrow_rate_daily_hist.append(borrow_rate_daily)
        borrow_rate_annual_hist.append(borrow_rate_daily * 252.0)

    out = data.iloc[: len(strategy_vals)].copy()
    out["Strategy_Value"] = strategy_vals
    out["Benchmark_Value"] = bench_vals
    out["Running_High_Close"] = data["Running_High_Close"].iloc[: len(out)]
    out["Off_High_Pct"] = off_high_hist[: len(out)]
    out["Allocation_Pct"] = np.array(alloc_hist) * 100.0
    out["Trade_Flag"] = trade_flag[: len(out)]
    out["Leverage_Flag"] = leverage_flag[: len(out)]
    out["Borrow_Rate_Daily"] = borrow_rate_daily_hist[: len(out)]
    out["Borrow_Rate_Annualized"] = borrow_rate_annual_hist[: len(out)]
    out["Strategy_Return"] = out["Strategy_Value"].pct_change().fillna(0.0)
    out["Benchmark_Return"] = out["Benchmark_Value"].pct_change().fillna(0.0)
    return out



def metrics(series):
    rets = series.pct_change().dropna()
    total_return = series.iloc[-1] / series.iloc[0] - 1.0
    years = max((series.index[-1] - series.index[0]).days / 365.25, 1 / 365.25)
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1.0
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0.0
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() and not np.isnan(rets.std()) else np.nan
    dd = series / series.cummax() - 1.0
    mdd = dd.min()
    return {
        "Ending Value": series.iloc[-1],
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
    }



def annual_snapshot(bt):
    annual = (
        bt[["Cash_Yield", "Nominal_Borrow_Rate_Annual", "Selected_Real_Rate", "Allocation_Pct", "Desired_Allocation_Pct"]]
        .groupby(bt.index.year)
        .agg(
            Avg_Cash_Yield=("Cash_Yield", "mean"),
            Avg_Nominal_Borrow_Rate=("Nominal_Borrow_Rate_Annual", "mean"),
            Avg_Selected_Real_Rate=("Selected_Real_Rate", "mean"),
            Avg_Allocation_Pct=("Allocation_Pct", "mean"),
            Avg_Desired_Allocation_Pct=("Desired_Allocation_Pct", "mean"),
        )
        .reset_index()
        .rename(columns={"index": "Year"})
    )
    return annual



def format_metrics_df(metrics_map):
    return pd.DataFrame(metrics_map).T


st.markdown("## Mode")
app_mode = st.radio(
    "Choose backtest type",
    ["Constant Leverage", "Buy the Dip off Highs", "Dividend Yield vs Rates"],
    horizontal=True,
)

st.markdown("## Common inputs")
row1 = st.columns(4)
with row1[0]:
    start_date = st.date_input(
        "Start date",
        value=date(2013, 1, 2),
        min_value=date(1934, 1, 1),
        max_value=date.today(),
        format="YYYY/MM/DD",
    )
with row1[1]:
    end_date = st.date_input(
        "End date",
        value=date.today(),
        min_value=date(1934, 1, 1),
        max_value=date.today(),
        format="YYYY/MM/DD",
    )
with row1[2]:
    initial_capital = st.number_input("Initial capital", min_value=1000, value=100000, step=1000)
with row1[3]:
    borrow_spread_bps = st.number_input("Borrow spread over Treasury (bps)", min_value=0, max_value=1000, value=150, step=5)

st.caption(
    "TB3MS is used for cash and nominal borrowing cost. The real-rate trigger uses the published FRED real-rate series: "
    "REAINTRATREARAT1YE or REAINTRATREARAT10Y. The research tab uses monthly FRED data and a built-in Shiller-derived monthly dividend-yield source."
)

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

if app_mode == "Constant Leverage":
    st.markdown("## Constant leverage inputs")
    rowc1 = st.columns(2)
    with rowc1[0]:
        reinvest_dividends = st.toggle(
            "Reinvest Dividends using SP500TR",
            value=False,
            help="Uses ^SP500TR as the return stream while keeping the strategy framework anchored to ^GSPC.",
            key="const_reinvest_toggle",
        )
    with rowc1[1]:
        trigger_mode = st.radio(
            "Trigger mode",
            ["Real Rate", "Dividend Yield Delta"],
            horizontal=True,
        )

    real_rate_source = st.radio(
            "Real-rate source",
            [
                "1-Year Real Rate (FRED REAINTRATREARAT1YE)",
                "10-Year Real Rate (FRED REAINTRATREARAT10Y)",
            ],
            horizontal=True,
        )

    uploaded_div_yield_file = None
    div_delta_rate_choice = None
    if trigger_mode == "Dividend Yield Delta":
        st.markdown("### Dividend yield delta input")
        uploaded_div_yield_file = st.file_uploader(
            "Upload dividend-yield file (.xlsx)",
            type=["xlsx", "xls"],
            key="const_div_yield_upload",
            help="File should contain Date and dividend yield columns, such as Date and Div Yield.",
        )
        div_delta_rate_choice = st.selectbox(
            "Delta comparison series",
            [
                "Dividend Yield - 1Y Nominal",
                "Dividend Yield - 1Y Real",
                "Dividend Yield - 10Y Nominal",
                "Dividend Yield - 10Y Real",
            ],
            index=1,
        )

    st.markdown("### Nominal throttle")
    rown = st.columns(2)
    with rown[0]:
        use_nominal_gate = st.checkbox("Use nominal borrow-rate throttle", value=True)
    with rown[1]:
        nominal_cap_pct = st.number_input(
            "Nominal borrow-rate cap (%)",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.25,
            disabled=not use_nominal_gate,
        )

    st.markdown("### Tiered leverage from real rate")
    rowt1 = st.columns(3)
    with rowt1[0]:
        base_alloc_pct = st.number_input("Base allocation (%)", min_value=0, max_value=300, value=100, step=5)
    with rowt1[1]:
        tier1_threshold_pct = st.number_input("Tier 1 real-rate threshold (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.25)
    with rowt1[2]:
        tier1_alloc_pct = st.number_input("Tier 1 allocation (%)", min_value=0, max_value=300, value=125, step=5)

    rowt2 = st.columns(3)
    with rowt2[0]:
        tier2_threshold_pct = st.number_input("Tier 2 real-rate threshold (%)", min_value=-10.0, max_value=10.0, value=1.0, step=0.25)
    with rowt2[1]:
        tier2_alloc_pct = st.number_input("Tier 2 allocation (%)", min_value=0, max_value=300, value=150, step=5)
    with rowt2[2]:
        st.write("")

   if app_mode == "Constant Leverage":
    if trigger_mode == "Dividend Yield Delta":
        st.caption(
            "Example: base 100%, then allocate 125% when dividend yield minus the selected rate is above Tier 1, and 150% when it is above Tier 2. "
            "If the nominal throttle is on and nominal borrowing cost is too high, the app caps exposure at 100% even if the dividend-yield delta rule wants leverage."
        )
    else:
        st.caption(
            "Example: base 100%, then allocate 125% when the selected real rate is below 2%, and 150% when it is below 1%. "
            "If the nominal throttle is on and nominal borrowing cost is too high, the app caps exposure at 100% even if the real-rate rule wants leverage."
        )

    if tier2_threshold_pct > tier1_threshold_pct:
        st.warning("Tier 2 threshold is usually lower than Tier 1 if you want deeper easing to trigger more leverage.")

    if st.button("Run constant leverage backtest", type="primary", use_container_width=True):
        try:
            df = load_gspc_data(start_date, end_date)
            df_run, used_tr = choose_return_stream(df, reinvest_dividends)

            if trigger_mode == "Dividend Yield Delta":
                if uploaded_div_yield_file is None:
                    raise ValueError("Please upload a dividend-yield file before running the Dividend Yield Delta backtest.")
                monthly_delta = build_dividend_delta_signal(start_date, end_date, uploaded_div_yield_file.getvalue())
                signal_col_map = {
                    "Dividend Yield - 1Y Nominal": "div_minus_gs1",
                    "Dividend Yield - 1Y Real": "div_minus_real_1y",
                    "Dividend Yield - 10Y Nominal": "div_minus_gs10",
                    "Dividend Yield - 10Y Real": "div_minus_real_10y",
                }
                df_run = map_dividend_delta_to_daily(df_run, monthly_delta, signal_col_map[div_delta_rate_choice])
            bt = run_constant_leverage_backtest(
                df=df_run,
                start=start_date,
                end=end_date,
                initial_capital=initial_capital,
                borrow_spread_bps=borrow_spread_bps,
                use_nominal_gate=use_nominal_gate,
                nominal_cap_pct=nominal_cap_pct,
                trigger_mode=trigger_mode,
                real_rate_source=real_rate_source,
                base_alloc_pct=base_alloc_pct,
                tier1_threshold_pct=tier1_threshold_pct,
                tier1_alloc_pct=tier1_alloc_pct,
                tier2_threshold_pct=tier2_threshold_pct,
                tier2_alloc_pct=tier2_alloc_pct,
            )

            bench_label = "Buy & Hold SP500TR" if used_tr else "Buy & Hold GSPC"
            results = format_metrics_df({"Strategy": metrics(bt["Strategy_Value"]), bench_label: metrics(bt["Benchmark_Value"])})

            st.markdown("## Results")
            st.caption(f"Return stream in use: {'^SP500TR' if used_tr else '^GSPC'}")
            st.dataframe(
                results.style.format(
                    {
                        "Ending Value": "${:,.0f}",
                        "Total Return": "{:.2%}",
                        "CAGR": "{:.2%}",
                        "Annualized Vol": "{:.2%}",
                        "Sharpe": "{:.2f}",
                        "Max Drawdown": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("## Equity curves")
            st.line_chart(bt[["Strategy_Value", "Benchmark_Value"]])

            st.markdown("## Allocation and trigger")
            trigger_cols = ["Allocation_Pct", "Desired_Allocation_Pct", "Selected_Real_Rate", "Nominal_Borrow_Rate_Annual"]
            if trigger_mode == "Dividend Yield Delta" and "Dividend_Delta_Signal" in bt.columns:
                trigger_cols = ["Allocation_Pct", "Desired_Allocation_Pct", "Dividend_Delta_Signal", "Nominal_Borrow_Rate_Annual"]
            st.line_chart(bt[trigger_cols])

            st.markdown("## Annual snapshot")
            annual = annual_snapshot(bt)
            st.dataframe(
                annual.style.format(
                    {
                        "Avg_Cash_Yield": "{:.2%}",
                        "Avg_Nominal_Borrow_Rate": "{:.2%}",
                        "Avg_Selected_Real_Rate": "{:.2%}",
                        "Avg_Allocation_Pct": "{:.1f}",
                        "Avg_Desired_Allocation_Pct": "{:.1f}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("## Daily output")
            st.dataframe(
                bt[
                    [
                        "Strategy_Value",
                        "Benchmark_Value",
                        "Allocation_Pct",
                        "Desired_Allocation_Pct",
                        "Rule_Label",
                        "Selected_Real_Rate",
                        "Nominal_Borrow_Rate_Annual",
                        "Nominal_Gate_Pass",
                        "Real_1Y_FRED",
                        "Real_10Y_FRED",
                        "GSPC_Close",
                        "SP500TR_Close",
                    ]
                ],
                use_container_width=True,
            )

        except Exception as e:
            st.exception(e)

elif app_mode == "Buy the Dip off Highs":
    st.markdown("## Dip strategy inputs")
    rowd1 = st.columns(3)
    with rowd1[0]:
        signal_mode = st.selectbox("Signal frequency", ["Daily", "Weekly"], index=1, key="dip_signal_mode")
    with rowd1[1]:
        reinvest_dividends = st.toggle(
            "Reinvest Dividends using SP500TR",
            value=False,
            key="dip_reinvest_toggle",
            help="Off-high signals stay based on ^GSPC closes and running highs; returns switch to ^SP500TR when enabled.",
        )
    with rowd1[2]:
        st.write("")

    rowd2 = st.columns(5)
    with rowd2[0]:
        off_high_t1 = st.number_input("Off-high threshold 1 (%)", value=5.0, step=0.5)
    with rowd2[1]:
        off_high_t2 = st.number_input("Off-high threshold 2 (%)", value=10.0, step=0.5)
    with rowd2[2]:
        alloc_low_dip = st.number_input("Allocation when shallow dip (%)", value=85, step=5)
    with rowd2[3]:
        alloc_mid_dip = st.number_input("Allocation at threshold 1 (%)", value=100, step=5)
    with rowd2[4]:
        alloc_deep_dip = st.number_input("Allocation at threshold 2 (%)", value=125, step=5)

    if st.button("Run buy-the-dip backtest", type="primary", use_container_width=True):
        try:
            df = load_gspc_data(start_date, end_date)
            df_run, used_tr = choose_return_stream(df, reinvest_dividends)

            if trigger_mode == "Dividend Yield Delta":
                if uploaded_div_yield_file is None:
                    raise ValueError("Please upload a dividend-yield file before running the Dividend Yield Delta backtest.")
                monthly_delta = build_dividend_delta_signal(start_date, end_date, uploaded_div_yield_file.getvalue())
                signal_col_map = {
                    "Dividend Yield - 1Y Nominal": "div_minus_gs1",
                    "Dividend Yield - 1Y Real": "div_minus_real_1y",
                    "Dividend Yield - 10Y Nominal": "div_minus_gs10",
                    "Dividend Yield - 10Y Real": "div_minus_real_10y",
                }
                df_run = map_dividend_delta_to_daily(df_run, monthly_delta, signal_col_map[div_delta_rate_choice])
            bt = run_buy_the_dip_backtest(
                df=df_run,
                start=start_date,
                end=end_date,
                signal_mode=signal_mode,
                t1=off_high_t1,
                t2=off_high_t2,
                a1=alloc_low_dip,
                a2=alloc_mid_dip,
                a3=alloc_deep_dip,
                initial_capital=initial_capital,
                borrow_spread_bps=borrow_spread_bps,
            )

            bench_label = "Buy & Hold SP500TR" if used_tr else "Buy & Hold GSPC"
            results = format_metrics_df({"Strategy": metrics(bt["Strategy_Value"]), bench_label: metrics(bt["Benchmark_Value"])})

            st.markdown("## Results")
            st.dataframe(
                results.style.format(
                    {
                        "Ending Value": "${:,.0f}",
                        "Total Return": "{:.2%}",
                        "CAGR": "{:.2%}",
                        "Annualized Vol": "{:.2%}",
                        "Sharpe": "{:.2f}",
                        "Max Drawdown": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("## Equity curves")
            st.line_chart(bt[["Strategy_Value", "Benchmark_Value"]])
            st.markdown("## Daily output")
            st.dataframe(bt, use_container_width=True)
        except Exception as e:
            st.exception(e)

elif app_mode == "Dividend Yield vs Rates":
    st.markdown("## Dividend Yield vs Rates")
    st.caption(
        "Monthly research view combining a built-in Shiller-derived S&P 500 dividend-yield series with key FRED rate series."
    )

    try:
        merged = load_dividend_yield_rates_data(start_date, end_date)
        merged = merged.loc[
            (merged["date"] >= pd.to_datetime(start_date).to_period("M").to_timestamp())
            & (merged["date"] <= pd.to_datetime(end_date).to_period("M").to_timestamp())
        ].copy()

        if merged.empty:
            st.warning("No monthly data found for the selected date range.")
        else:
            min_date = merged["date"].min().date()
            max_date = merged["date"].max().date()

            filter_cols = st.columns(3)
            with filter_cols[0]:
                chart_start = st.date_input(
                    "Chart start",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="div_rates_chart_start",
                )
            with filter_cols[1]:
                chart_end = st.date_input(
                    "Chart end",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="div_rates_chart_end",
                )
            with filter_cols[2]:
                series_options = [
                    "sp500_dividend_yield",
                    "fedfunds",
                    "gs1",
                    "gs10",
                    "real_1y",
                    "real_10y",
                ]
                selected_series = st.multiselect(
                    "Series to plot",
                    options=series_options,
                    default=["sp500_dividend_yield", "fedfunds", "gs10", "real_10y"],
                    key="div_rates_series_multiselect",
                )

            chart_df = merged.loc[
                (merged["date"] >= pd.to_datetime(chart_start).to_period("M").to_timestamp())
                & (merged["date"] <= pd.to_datetime(chart_end).to_period("M").to_timestamp())
            ].copy()

            st.markdown("## Chart")
            if selected_series:
                fig = make_dividend_yield_rates_chart(chart_df, selected_series)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Select at least one series to display the chart.")

            st.markdown("## Merged monthly dataset")
            st.dataframe(chart_df, use_container_width=True)

            csv_bytes = chart_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download merged CSV",
                data=csv_bytes,
                file_name="dividend_yield_vs_rates.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.exception(e)
