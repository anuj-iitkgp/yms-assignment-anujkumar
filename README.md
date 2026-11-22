# YMS Financials Assignment

## Will a Nifty-50 stock move more than 2% today?

For every trading day and every Nifty-50 stock, predict **before the market opens** whether the stock
will trade more than 2% away from its previous close at any point during the day:

```
moved = max(high - pclose, pclose - low) / pclose > 2%
```

`high`/`low` are that day's high and low and `pclose` is the previous close. In other words, the stock
leaves the band `(pclose × 0.98, pclose × 1.02)` at least once.

## What you get

| Path | Contents |
|---|---|
| `data/historical_info/<YYYYMMDD>_n50.info.csv` | One file per **training** date (97 dates, 2026-03-04 .. 2026-10-30), 50 rows (one per stock), known at that day's open |
| `data/train_labels.csv` | The answer for every training (date, name): `moved`, plus `max_move_pct`, `up_move_pct`, `down_move_pct` |
| `data/all_dates.txt` | All 138 dates: the training dates **and** the 41 hidden test dates, which are drawn at random from the same period |
| `data/n50_names.txt` | The 50 stock symbols |
| `predict.py` | The program you submit (a working example to replace) |
| `prepare_user_historical_info.py` | Builds your extra data file (a template to replace) |
| `test_predictions.py` | The official scorer, so you can score yourself on the training dates |

### Columns of an info file (everything is known before 09:15 on that date)

| Column | Meaning |
|---|---|
| `date`, `name` | Trading day (YYYYMMDD), NSE symbol |
| `pclose` | Previous trading day's close (the reference price for the 2% band) |
| `phi`, `plo` | Previous trading day's high and low (blank if that day's data is incomplete) |
| `pvolume` | Previous day's cash-market traded value, in rupees |
| `mddv_20` | Median daily traded value over the last 20 days, in rupees |
| `vix_asof` | The date the VIX columns below were computed on (the latest day before `date`) |
| `vix` | The stock's option-implied volatility (annualised %), averaged over `vix_asof`'s session. See below |
| `vix_1m`, `vix_3m`, `vix_6m`, `vix_1y` | Averages of the daily `vix` over the last 30 / 90 / 180 / 365 calendar days up to `vix_asof` |
| `vix_zscore_smooth`, `vix_zscore_raw` | How high the stock's `vix` is compared with the typical volatility of all ~224 F&O stocks. See below |

### Understanding the volatility columns

**`vix`: the stock's own "fear gauge".** It is the volatility the options market expects for that stock
over the next ~30 days, worked out from the prices of its options (the same method as the India VIX /
CBOE VIX, applied to one stock). It is an annualised percentage, computed every minute from 09:30 to
15:30 and averaged over the day. Higher means options are pricing bigger moves.

A useful rule of thumb turns it into an expected **daily** move: `vix / sqrt(252)` is roughly one
standard deviation of a day's close-to-close move. For example `vix = 32` → about `32 / 15.9 ≈ 2.0%` a
day. The daily high-low range is usually larger than the close-to-close move.

**`vix_1m`, `vix_3m`, `vix_6m`, `vix_1y`: its recent history.** Averages of the daily `vix` over the
last 30 / 90 / 180 / 365 calendar days. The data starts on 2026-03-02, so early on these windows hold
only the days available: `vix_1y` always covers everything since 2026-03-02, and `vix_6m` equals it
until 2026-09-01. Comparing `vix` with
`vix_1m` shows whether a stock's volatility is rising or falling.

**`vix_zscore_smooth`: how volatile this stock is compared with the whole market.**

```
vix_zscore_smooth = (vix - M) / S
```

- `M` is the typical volatility level: the average, over ~224 F&O stocks, of each stock's own
  3-month average `vix` (for example M ≈ 28).
- `S` is the typical spread: it combines how much each stock's `vix` varies over 3 months with how
  much the stocks differ from each other (for example S ≈ 7).
- `M` and `S` are the same for every stock on a given date, so the z-score ranks stocks against each
  other: `0` = an average F&O stock, `+1` = one spread above it (≈ 35 when M ≈ 28, S ≈ 7), `-1` = calmer
  than usual.

**`vix_zscore_raw`** is the same calculation on the unsmoothed minute-by-minute VIX (`vix` and
`vix_zscore_smooth` use a lightly smoothed series). The two are usually close; the raw one reacts
more to short spikes.

On some training dates `pclose`, `phi`, `plo` and the VIX columns are blank: the day before was a hidden
test day, and those values would give its answer away. Their labels in `train_labels.csv` are complete.
During evaluation, every info file you are given is complete.

## What you submit

1. **`predict.py`**: run once per date as

   ```
   python predict.py --date YYYYMMDD --historical-info DIR --user-historical-info FILE --out-dir DIR
   ```

   It must write `<out-dir>/<YYYYMMDD>.predictions.csv` with columns `name,prediction`, one row per stock in
   that day's info file, `prediction` = `True` if you expect a move of more than 2%, otherwise `False`.
2. **`prepare_user_historical_info.py`** and the **`user_historical_info.csv`** it produces: any extra data
   you want (market data APIs, index levels, global markets, …). Columns `date,name,<your features>`.
   Cover **every date in `data/all_dates.txt`**.
3. Any helper modules and model files your `predict.py` needs (not data files: see below).

Python 3 is assumed. List any packages you need in `requirements.txt`.

## How you are evaluated

The evaluator runs your `predict.py` on each hidden test date, one date at a time, in a fresh folder that
contains only:
- `historical_info/` with the info files for every date **up to and including** the date being predicted
  (training dates and earlier test dates, never later ones);
- your `user_historical_info.csv` with rows dated **after** that date removed.

CSV/feather/parquet/pickle/zip files inside your submission are not copied into that folder, so put
learned parameters in code or a small `.json`/`.txt` file.

Scores over all test (date, stock) rows, where a "positive" is a predicted move (`True`):
- **precision** = TP / (TP + FP): of the moves you predicted, how many happened
- **recall** = TP / (TP + FN): of the moves that happened, how many you predicted
- **combined score = 0.75 × precision + 0.25 × recall**
- to pass, **precision must be above 60%**; candidates are ranked by the combined score

Predicting nothing scores 0. A missing prediction file or row counts as wrong (a missed move, or a false
alarm). The test dates are a random 30% of the same period, so their move rate is similar but not equal
(about 34% of rows move on the test dates, 41% on the training dates).

Check yourself on the training dates:

```
python prepare_user_historical_info.py
for d in $(ls data/historical_info | cut -c1-8); do
  python predict.py --date $d --historical-info data/historical_info \
                    --user-historical-info user_historical_info.csv --out-dir predictions
done
python test_predictions.py predictions --labels data/train_labels.csv
```

## Rules

- **No look-ahead.** A prediction for day D may only use information public before 09:15 on day D. A
  row of `user_historical_info.csv` dated D must follow the same rule (for example the previous day's
  close is fine, day D's own high or close is not).
- When predicting day D, `predict.py` may read only the files it is given.
- Explain your approach in a short `NOTES.md`: features, model, and how you tuned it without using the
  test dates.
