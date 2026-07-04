from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# ReCAP gets DOW30 tickers from finrl.config_tickers.DOW_30_TICKER.
# This fallback mirrors the 29-ticker DOW30 universe commonly used by FinRL/ReCAP
# for the 2008-2025 window. DOW is excluded because it does not have full history
# back to 2008.
FALLBACK_DOW30_TICKERS = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
]

BASE_COLUMNS = ["open", "high", "low", "close", "volume"]
FEATURE_COLUMNS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "zd_5",
    "zd_10",
    "zd_15",
    "zd_20",
    "zd_25",
    "zd_30",
    "zopen",
    "zhigh",
    "zlow",
    "zadjcp",
    "zclose",
    "vix",
    "turbulence",
    *BASE_COLUMNS,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download DOW30 daily Yahoo Finance data in a ReCAP-style format."
    )
    parser.add_argument("--start-date", default="2008-05-01")
    # yfinance treats end as exclusive. 2025-04-30 includes data through 2025-04-29.
    parser.add_argument("--end-date", default="2025-04-30")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "dow30"),
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--raw-only", action="store_true", help="Skip ReCAP-style feature engineering.")
    parser.add_argument(
        "--disable-finrl-tickers",
        action="store_true",
        help="Use the built-in fallback ticker list even if FinRL is installed.",
    )
    return parser


def get_dow30_tickers(disable_finrl: bool = False) -> list[str]:
    if not disable_finrl:
        try:
            from finrl.config_tickers import DOW_30_TICKER

            tickers = list(DOW_30_TICKER)
            if tickers:
                return tickers
        except Exception:
            pass
    return FALLBACK_DOW30_TICKERS


def get_yfinance():
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("Missing dependency: yfinance. Install with `pip install yfinance`.") from exc
    return yf


def download_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    batch_size: int = 50,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start : start + batch_size]
        for ticker in batch:
            try:
                frames.append(download_one_yahoo_chart(ticker=ticker, start_date=start_date, end_date=end_date))
            except Exception as chart_error:
                try:
                    yf = get_yfinance()
                    raw = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                    )
                    frames.extend(_normalize_download_frame(raw=raw, requested_tickers=[ticker]))
                except Exception:
                    print(f"Warning: failed to download {ticker}: {chart_error}")

    if not frames:
        raise ValueError("No price data was downloaded.")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "tic"]).reset_index(drop=True)
    available_tickers = sorted(data["tic"].unique().tolist())
    missing_tickers = sorted(set(tickers) - set(available_tickers))
    if missing_tickers:
        print(f"Warning: dropping tickers with no downloaded data: {', '.join(missing_tickers)}")
    return keep_complete_trading_dates(data, available_tickers)


def download_one_yahoo_chart(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    period1 = to_unix_timestamp(start_date)
    period2 = to_unix_timestamp(end_date)
    encoded = quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    payload = json.loads(urlopen(request, timeout=30).read().decode("utf-8"))
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(chart["error"])
    results = chart.get("result") or []
    if not results:
        raise ValueError("empty Yahoo chart result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjclose_data = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if not timestamps or not quote_data:
        raise ValueError("missing Yahoo chart price data")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s").tz_localize("UTC").tz_convert(None).normalize(),
            "tic": ticker,
            "open": quote_data.get("open"),
            "high": quote_data.get("high"),
            "low": quote_data.get("low"),
            "close": quote_data.get("close"),
            "volume": quote_data.get("volume"),
        }
    )
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    if adjclose_data:
        frame["adjcp"] = pd.Series(adjclose_data, index=frame.index)
        ratio = frame["adjcp"] / np.clip(frame["close"], 1e-12, None)
        for column in ["open", "high", "low", "close"]:
            frame[column] = frame[column] * ratio
    return frame[["date", "tic", "open", "high", "low", "close", "volume"]]


def to_unix_timestamp(date_string: str) -> int:
    date_value = dt.datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(date_value.timestamp())


def add_recap_style_features(data: pd.DataFrame, use_vix: bool = True) -> pd.DataFrame:
    df = data.copy()
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)
    grouped = df.groupby("tic", group_keys=False)

    df["return_1"] = grouped["close"].pct_change().fillna(0.0)
    df["return_5"] = grouped["close"].pct_change(5).fillna(0.0)
    df["adjcp"] = pd.to_numeric(df.get("adjcp", df["close"]), errors="coerce").fillna(df["close"])

    df["close_30_sma"] = grouped["close"].transform(lambda values: values.rolling(30, min_periods=1).mean())
    df["close_60_sma"] = grouped["close"].transform(lambda values: values.rolling(60, min_periods=1).mean())
    df["macd"] = grouped["adjcp"].transform(calculate_macd)
    df["boll_ub"], df["boll_lb"] = calculate_bollinger_bands(grouped["adjcp"])
    df["rsi_30"] = grouped["adjcp"].transform(lambda values: calculate_rsi(values, window=30))

    grouped_prices = df.groupby("tic")[["high", "low", "close"]]
    df["cci_30"] = grouped_prices.apply(calculate_cci_30).reset_index(level=0, drop=True)
    df["dx_30"] = grouped_prices.apply(calculate_dx_30).reset_index(level=0, drop=True)

    for period in [5, 10, 15, 20, 25, 30]:
        df[f"zd_{period}"] = grouped["adjcp"].pct_change(period).fillna(0.0)

    close_scale = np.clip(df["close"], 1e-12, None)
    df["zopen"] = df["open"] / close_scale - 1.0
    df["zhigh"] = df["high"] / close_scale - 1.0
    df["zlow"] = df["low"] / close_scale - 1.0
    df["zclose"] = df["return_1"]
    df["zadjcp"] = grouped["adjcp"].pct_change().fillna(0.0)

    if use_vix:
        vix_frame = download_vix_frame(
            start_date=pd.Timestamp(df["date"].min()),
            end_date=pd.Timestamp(df["date"].max()) + pd.Timedelta(days=1),
        )
        df = df.merge(vix_frame, on="date", how="left")
        df["vix"] = pd.to_numeric(df["vix"], errors="coerce").ffill().bfill()
    else:
        df["vix"] = np.nan

    turbulence = calculate_turbulence(df)
    df = df.merge(turbulence, on="date", how="left")

    numeric_columns = [
        "adjcp",
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30",
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "vix",
        "turbulence",
        "return_1",
        "return_5",
    ]
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return keep_complete_trading_dates(df, sorted(df["tic"].unique().tolist()))


def _normalize_download_frame(raw: pd.DataFrame, requested_tickers: list[str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    if raw.empty:
        return frames

    if isinstance(raw.columns, pd.MultiIndex):
        level_zero = set(raw.columns.get_level_values(0))
        level_one = set(raw.columns.get_level_values(1))
        ticker_first = any(ticker in level_zero for ticker in requested_tickers)

        for ticker in requested_tickers:
            if ticker_first:
                if ticker not in level_zero:
                    continue
                ticker_frame = raw[ticker].copy().reset_index()
            else:
                if ticker not in level_one:
                    continue
                ticker_frame = raw.xs(ticker, axis=1, level=1).copy().reset_index()
            ticker_frame.columns = [str(column).lower() for column in ticker_frame.columns]
            ticker_frame["tic"] = ticker
            frames.append(ticker_frame[["date", "tic", *BASE_COLUMNS]])
        return frames

    ticker = requested_tickers[0]
    single_frame = raw.copy().reset_index()
    single_frame.columns = [str(column).lower() for column in single_frame.columns]
    single_frame["tic"] = ticker
    frames.append(single_frame[["date", "tic", *BASE_COLUMNS]])
    return frames


def keep_complete_trading_dates(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    expected_count = len(tickers)
    counts = Counter(data["date"])
    valid_dates = {date for date, count in counts.items() if count == expected_count}
    filtered = data[data["date"].isin(valid_dates)].copy()
    filtered = filtered.sort_values(["date", "tic"]).reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No complete trading dates remain after alignment.")
    return filtered


def calculate_macd(prices: pd.Series) -> pd.Series:
    ema_short = prices.ewm(span=12, adjust=False).mean()
    ema_long = prices.ewm(span=26, adjust=False).mean()
    return (ema_short - ema_long).fillna(0.0)


def calculate_bollinger_bands(grouped: pd.core.groupby.generic.SeriesGroupBy) -> tuple[pd.Series, pd.Series]:
    rolling_mean = grouped.transform(lambda values: values.rolling(window=20, min_periods=1).mean())
    rolling_std = grouped.transform(lambda values: values.rolling(window=20, min_periods=1).std().fillna(0.0))
    return (rolling_mean + 2.0 * rolling_std).fillna(0.0), (rolling_mean - 2.0 * rolling_std).fillna(0.0)


def calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    delta = close_prices.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.rolling(window=window, min_periods=window).mean()
    average_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = average_gain / np.clip(average_loss, 1e-12, None)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def calculate_cci_30(frame: pd.DataFrame) -> pd.Series:
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    rolling_mean = typical_price.rolling(window=30, min_periods=1).mean()
    mean_deviation = typical_price.rolling(window=30, min_periods=1).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))),
        raw=True,
    )
    cci = (typical_price - rolling_mean) / np.clip(0.015 * mean_deviation, 1e-12, None)
    return cci.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def calculate_dx_30(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    up_move = high.diff().fillna(0.0)
    down_move = -low.diff().fillna(0.0)
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1).fillna(0.0)
    atr = true_range.rolling(window=30, min_periods=1).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=frame.index).rolling(window=30, min_periods=1).mean() / np.clip(atr, 1e-12, None)
    minus_di = 100.0 * pd.Series(minus_dm, index=frame.index).rolling(window=30, min_periods=1).mean() / np.clip(atr, 1e-12, None)
    dx = 100.0 * (plus_di - minus_di).abs() / np.clip(plus_di + minus_di, 1e-12, None)
    return dx.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def calculate_turbulence(data: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    returns = data.pivot(index="date", columns="tic", values="close").pct_change().fillna(0.0)
    turbulence_values = np.zeros(len(returns), dtype=np.float32)
    for index in range(len(returns)):
        start = max(0, index - lookback)
        history = returns.iloc[start:index]
        if history.shape[0] < 5:
            continue
        mean = history.mean(axis=0).to_numpy(dtype=np.float64)
        cov = history.cov().to_numpy(dtype=np.float64)
        cov += np.eye(cov.shape[0], dtype=np.float64) * 1e-6
        current = returns.iloc[index].to_numpy(dtype=np.float64)
        diff = current - mean
        turbulence_values[index] = float(diff @ np.linalg.pinv(cov) @ diff)
    return pd.DataFrame({"date": returns.index, "turbulence": turbulence_values})


def download_vix_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        return download_yahoo_close_frame(
            ticker="^VIX",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            close_column="vix",
        )
    except Exception:
        return pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="B"), "vix": np.nan})


def download_yahoo_close_frame(ticker: str, start_date: str, end_date: str, close_column: str) -> pd.DataFrame:
    period1 = to_unix_timestamp(start_date)
    period2 = to_unix_timestamp(end_date)
    encoded = quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    payload = json.loads(urlopen(request, timeout=30).read().decode("utf-8"))
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise ValueError(chart["error"])
    result = (chart.get("result") or [])[0]
    quote_data = (result.get("indicators", {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result.get("timestamp") or [], unit="s").tz_localize("UTC").tz_convert(None).normalize(),
            close_column: quote_data.get("close"),
        }
    )
    return frame.dropna(subset=[close_column]).copy()


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = get_dow30_tickers(disable_finrl=args.disable_finrl_tickers)
    raw_data = download_price_data(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
    )

    raw_path = output_dir / "Yahoo_DOW_30_TICKER_1D_cache.csv"
    tickers_path = output_dir / "DOW30_tickers.txt"
    raw_data.to_csv(raw_path, index=False)
    available_tickers = sorted(raw_data["tic"].unique().tolist())
    tickers_path.write_text("\n".join(available_tickers) + "\n", encoding="utf-8")

    print(f"Saved raw DOW30 cache: {raw_path}")
    print(f"Requested tickers: {len(tickers)}")
    print(f"Available tickers: {len(available_tickers)}")
    print(f"Trading dates: {raw_data['date'].nunique()}")
    print(f"Rows: {len(raw_data)}")

    if args.raw_only:
        return

    feature_data = add_recap_style_features(raw_data, use_vix=True)
    feature_path = output_dir / "DOW30_recap_features.csv"
    feature_data.to_csv(feature_path, index=False)
    print(f"Saved ReCAP-style features: {feature_path}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()
