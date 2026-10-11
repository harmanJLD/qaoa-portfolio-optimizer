"""Build data/assets.csv from real public market prices. Run this on your own computer,
not on Qollab (it needs a network connection and the yfinance package).

    pip install yfinance pandas numpy
    python tools/make_dataset.py                      # default: 8 SPDR sector ETFs, 3 years
    python tools/make_dataset.py --tickers AAPL MSFT JPM XOM JNJ NEE PG CAT --years 5

Then upload the new data/assets.csv to the Qollab project. Keep 8 or fewer assets
(8 qubits) so the local study stays fast in the browser.
"""
import argparse
import numpy as np
import pandas as pd

DEFAULT = {  # SPDR sector ETFs, labelled by sector
    "XLK": "TECH", "XLV": "HLTH", "XLE": "ENRG", "XLF": "FIN",
    "XLU": "UTIL", "XLP": "STAP", "XLI": "INDU", "XLB": "MATL",
}


def build(prices: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Annualized mean return, volatility and correlation from daily closing prices."""
    daily = np.log(prices).diff().dropna()
    mu = daily.mean() * 252
    vol = daily.std() * np.sqrt(252)
    corr = daily.corr()
    out = pd.DataFrame({"asset": labels, "expected_return": mu.values.round(4),
                        "volatility": vol.values.round(4)})
    for j, lab in enumerate(labels):
        out["corr_" + lab] = corr.iloc[:, j].values.round(4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=list(DEFAULT))
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--out", default="data/assets.csv")
    args = ap.parse_args()

    import tempfile
    import time
    import yfinance as yf
    # A private cache folder avoids yfinance's "database is locked" error
    yf.set_tz_cache_location(tempfile.mkdtemp())

    series = {}
    for t in args.tickers:
        for attempt in range(4):
            try:
                hist = yf.Ticker(t).history(period=f"{args.years}y", auto_adjust=True)
                if len(hist) > 100:
                    series[t] = hist["Close"]
                    print(f"  {t}: {len(hist)} days")
                    break
            except Exception as e:
                print(f"  {t}: attempt {attempt + 1} failed ({e})")
            time.sleep(2)
        else:
            raise SystemExit(f"Could not download {t}. Nothing was saved. Try again in a minute.")

    prices = pd.DataFrame(series)[args.tickers].dropna()
    if len(prices) < 100:
        raise SystemExit("Too few overlapping trading days. Nothing was saved.")
    labels = [DEFAULT.get(t, t) for t in args.tickers]
    df = build(prices, labels)
    if df.isna().any().any():
        raise SystemExit("Some values came out empty. Nothing was saved.")
    df.to_csv(args.out, index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nSaved {args.out}: {len(prices)} trading days, "
          f"{prices.index[0].date()} to {prices.index[-1].date()}")


if __name__ == "__main__":
    main()
