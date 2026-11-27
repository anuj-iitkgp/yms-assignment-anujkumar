#!/usr/bin/env python3
"""Build user_historical_info.csv with strictly pre-09:15 IST market context features.

  python prepare_user_historical_info.py --dates data/all_dates.txt --names data/n50_names.txt \
                                         --out user_historical_info.csv

For each date D in data/all_dates.txt, fetches historical daily closes for ^NSEI (Nifty 50)
and ^INDIAVIX (India VIX) and computes features using ONLY trading days strictly before D
(date_ist < D), ensuring zero look-ahead bias before 09:15 IST on day D.
"""
import argparse
import csv
import datetime
import json
import math
import urllib.parse
import urllib.request


def fetch_yahoo_daily(symbol, period1=1769904000, period2=1794000000):
    """Fetch daily bars (date_ist, close) from Yahoo Finance chart endpoint."""
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as resp:
        payload = json.loads(resp.read().decode('utf-8'))
    res = payload['chart']['result'][0]
    timestamps = res['timestamp']
    closes = res['indicators']['quote'][0]['close']
    ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    bars = []
    for ts, close in zip(timestamps, closes):
        if close is None or not math.isfinite(close) or close <= 0:
            continue
        dt_ist = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).astimezone(ist)
        bars.append((dt_ist.strftime('%Y%m%d'), float(close)))
    bars.sort(key=lambda x: x[0])
    # Deduplicate by date_ist keeping latest
    dedup = {}
    for d, c in bars:
        dedup[d] = c
    return sorted(dedup.items())


def compute_nsei_rv5d(nsei_bars):
    """Compute annualised 5-day close-to-close realized volatility (%) per date_ist."""
    out = []
    rets = []
    for i, (d, c) in enumerate(nsei_bars):
        if i == 0:
            out.append((d, None))
            continue
        prev_c = nsei_bars[i - 1][1]
        r = (c - prev_c) / prev_c
        rets.append(r)
        if len(rets) >= 5:
            win = rets[-5:]
            mean_r = sum(win) / 5.0
            var_r = sum((x - mean_r) ** 2 for x in win) / 4.0
            rv5d = math.sqrt(var_r) * math.sqrt(252.0) * 100.0
            out.append((d, rv5d))
        else:
            out.append((d, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', default='data/all_dates.txt')
    ap.add_argument('--names', default='data/n50_names.txt')
    ap.add_argument('--out', default='user_historical_info.csv')
    a = ap.parse_args()

    dates = open(a.dates).read().split()
    names = open(a.names).read().split()

    nsei_bars = fetch_yahoo_daily('^NSEI')
    ivix_bars = fetch_yahoo_daily('^INDIAVIX')
    nsei_rv = compute_nsei_rv5d(nsei_bars)

    with open(a.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date', 'name', 'weekday', 'indiavix', 'log_indiavix', 'nsei_rv5d'])
        for d in dates:
            weekday = datetime.datetime.strptime(d, '%Y%m%d').weekday()
            # Strictly earlier than d (closed at 15:30 IST on trading day < d)
            past_vix = [c for dt, c in ivix_bars if dt < d]
            past_rv = [rv for dt, rv in nsei_rv if dt < d and rv is not None]
            indiavix = past_vix[-1] if past_vix else ''
            log_indiavix = math.log(past_vix[-1]) if past_vix else ''
            nsei_rv5d = past_rv[-1] if past_rv else ''
            for n in names:
                w.writerow([d, n, weekday, indiavix, log_indiavix, nsei_rv5d])

    print('wrote %s: %d dates x %d names' % (a.out, len(dates), len(names)))


if __name__ == '__main__':
    main()
