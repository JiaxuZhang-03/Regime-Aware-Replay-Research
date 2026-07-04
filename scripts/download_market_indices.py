from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_SYMBOLS = [
    "SPY",
    "DIA",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
    "^GSPC",
    "^DJI",
    "^IXIC",
    "^VIX",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download daily market index/ETF data for regime detection.")
    parser.add_argument("--start-date", default="2008-06-01")
    parser.add_argument("--end-date", default="2026-06-01")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Yahoo symbols to download. End date is exclusive, Yahoo-style.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "market_indices_20080601_20260531"),
    )
    return parser


def to_unix_timestamp(date_string: str) -> int:
    date_value = dt.datetime.strptime(date_string, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(date_value.timestamp())


def download_one_yahoo_chart(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    period1 = to_unix_timestamp(start_date)
    period2 = to_unix_timestamp(end_date)
    encoded = quote(symbol, safe="")
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
    adjclose_data = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result.get("timestamp") or [], unit="s").tz_localize("UTC").tz_convert(None).normalize(),
            "symbol": symbol,
            "open": quote_data.get("open"),
            "high": quote_data.get("high"),
            "low": quote_data.get("low"),
            "close": quote_data.get("close"),
            "volume": quote_data.get("volume"),
        }
    )
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    if adjclose_data and len(adjclose_data) == len(frame):
        frame["adjclose"] = pd.Series(adjclose_data, index=frame.index)
    else:
        frame["adjclose"] = frame["close"]
    return frame[["date", "symbol", "open", "high", "low", "close", "adjclose", "volume"]]


def add_regime_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    grouped = df.groupby("symbol", group_keys=False)
    df["ret_1d"] = grouped["adjclose"].pct_change()
    df["ret_5d"] = grouped["adjclose"].pct_change(5)
    df["ret_20d"] = grouped["adjclose"].pct_change(20)
    df["ret_60d"] = grouped["adjclose"].pct_change(60)
    df["vol_20d"] = grouped["ret_1d"].transform(lambda s: s.rolling(20, min_periods=10).std() * (252 ** 0.5))
    df["vol_60d"] = grouped["ret_1d"].transform(lambda s: s.rolling(60, min_periods=20).std() * (252 ** 0.5))
    df["ma_20"] = grouped["adjclose"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    df["ma_60"] = grouped["adjclose"].transform(lambda s: s.rolling(60, min_periods=1).mean())
    df["ma_200"] = grouped["adjclose"].transform(lambda s: s.rolling(200, min_periods=20).mean())
    df["trend_20_60"] = df["ma_20"] / df["ma_60"] - 1.0
    df["trend_price_200"] = df["adjclose"] / df["ma_200"] - 1.0
    feature_cols = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "ret_60d",
        "vol_20d",
        "vol_60d",
        "trend_20_60",
        "trend_price_200",
    ]
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def build_wide_regime_table(features: pd.DataFrame) -> pd.DataFrame:
    selected = features[["date", "symbol", "adjclose", "ret_20d", "ret_60d", "vol_20d", "trend_20_60", "trend_price_200"]]
    wide = selected.pivot(index="date", columns="symbol")
    wide.columns = [f"{field}_{symbol.replace('^', '')}" for field, symbol in wide.columns]
    wide = wide.reset_index()
    return wide.sort_values("date").reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    failures = []
    for symbol in args.symbols:
        try:
            frame = download_one_yahoo_chart(symbol=symbol, start_date=args.start_date, end_date=args.end_date)
            frames.append(frame)
            print(f"Downloaded {symbol}: {len(frame)} rows")
        except Exception as exc:
            failures.append((symbol, str(exc)))
            print(f"Warning: failed to download {symbol}: {exc}")

    if not frames:
        raise ValueError("No index data was downloaded.")

    raw = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    features = add_regime_features(raw)
    wide = build_wide_regime_table(features)

    raw.to_csv(output_dir / "market_indices_raw.csv", index=False)
    features.to_csv(output_dir / "market_indices_features_long.csv", index=False)
    wide.to_csv(output_dir / "market_regime_features_wide.csv", index=False)
    (output_dir / "symbols.txt").write_text("\n".join(sorted(raw["symbol"].unique())) + "\n", encoding="utf-8")

    if failures:
        (output_dir / "download_failures.txt").write_text(
            "\n".join(f"{symbol}: {reason}" for symbol, reason in failures) + "\n",
            encoding="utf-8",
        )

    print(f"Saved raw data: {output_dir / 'market_indices_raw.csv'}")
    print(f"Saved long features: {output_dir / 'market_indices_features_long.csv'}")
    print(f"Saved wide regime features: {output_dir / 'market_regime_features_wide.csv'}")
    print(f"Symbols downloaded: {raw['symbol'].nunique()}")
    print(f"Date range: {raw['date'].min()} to {raw['date'].max()}")


if __name__ == "__main__":
    main()
