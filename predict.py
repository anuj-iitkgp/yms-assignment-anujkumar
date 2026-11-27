#!/usr/bin/env python3
"""Predict, before market open, whether each Nifty-50 stock will move > 2% today.

Interface:
  python predict.py --date YYYYMMDD --historical-info DIR --user-historical-info FILE --out-dir DIR
"""
import argparse
import csv
import datetime
import json
import math
import os


SQRT_252 = math.sqrt(252.0)
SQRT_2 = math.sqrt(2.0)
LOG_2 = math.log(2.0)
PRIOR_RATIO = 1.28
PRIOR_WEIGHT = 8.0


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_linear_model(x_vec, model_dict, is_logreg=False):
    medians = model_dict['medians']
    means = model_dict['means']
    scales = model_dict['scales']
    coef = model_dict['coef']
    intercept = model_dict['intercept']

    z = intercept
    for i, val in enumerate(x_vec):
        v = medians[i] if val is None else val
        sc = scales[i] if scales[i] != 0 else 1.0
        z += coef[i] * ((v - means[i]) / sc)
    return sigmoid(z) if is_logreg else z


def load_user_info(path, target_date):
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('date') == target_date:
                out[r['name'].strip()] = r
    return out


def compute_past_stock_stats(hist_dir, target_date):
    """Compute strictly causal (< target_date) stock volatility statistics from historical_info."""
    target_dt = datetime.datetime.strptime(target_date, '%Y%m%d')
    files = sorted(
        fn for fn in os.listdir(hist_dir)
        if fn.endswith('_n50.info.csv') and fn[:8] < target_date
    )

    past_rng_14 = {}
    past_rng_all = {}
    past_iv_14 = {}
    past_iv_all = {}
    past_ratios = {}

    for fn in files:
        d_str = fn[:8]
        days_ago = (target_dt - datetime.datetime.strptime(d_str, '%Y%m%d')).days
        with open(os.path.join(hist_dir, fn), newline='') as f:
            for r in csv.DictReader(f):
                name = r['name'].strip()
                pc = num(r.get('pclose'))
                hi = num(r.get('phi'))
                lo = num(r.get('plo'))
                vix = num(r.get('vix'))

                iv = vix / 100.0 / SQRT_252 if (vix is not None and vix > 0) else None
                rng = (hi - lo) / pc if (pc and pc > 0 and hi is not None and lo is not None and hi >= lo) else None

                if rng is not None:
                    past_rng_all.setdefault(name, []).append(rng)
                    if days_ago <= 14:
                        past_rng_14.setdefault(name, []).append(rng)
                if iv is not None:
                    past_iv_all.setdefault(name, []).append(iv)
                    if days_ago <= 14:
                        past_iv_14.setdefault(name, []).append(iv)
                if rng is not None and iv is not None and iv > 0:
                    past_ratios.setdefault(name, []).append(rng / iv)

    stats = {}
    all_names = set(past_rng_all) | set(past_iv_all) | set(past_ratios)
    for name in all_names:
        r14 = past_rng_14.get(name)
        rall = past_rng_all.get(name)
        i14 = past_iv_14.get(name)
        iall = past_iv_all.get(name)
        rats = past_ratios.get(name, [])

        hr14 = (sum(r14) / len(r14)) if r14 else ((sum(rall) / len(rall)) if rall else None)
        hi14 = (sum(i14) / len(i14)) if i14 else ((sum(iall) / len(iall)) if iall else None)
        sr = (sum(rats) + PRIOR_WEIGHT * PRIOR_RATIO) / (len(rats) + PRIOR_WEIGHT)
        stats[name] = (hr14, hi14, sr)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    ap.add_argument('--historical-info', required=True)
    ap.add_argument('--user-historical-info', default=None)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, 'model_params.json')) as f:
        params = json.load(f)

    today_path = os.path.join(a.historical_info, a.date + '_n50.info.csv')
    with open(today_path, newline='') as f:
        today_rows = list(csv.DictReader(f))

    user_today = load_user_info(a.user_historical_info, a.date)
    past_stats = compute_past_stock_stats(a.historical_info, a.date)

    # Cross-sectional mean VIX on target date (pre-open across the N50 stocks)
    valid_vix = [num(r.get('vix')) for r in today_rows]
    valid_vix = [v for v in valid_vix if v is not None and v > 0]
    mkt_vix_mean = sum(valid_vix) / len(valid_vix) if valid_vix else None

    scale = params['log_move_scale']
    thr = params['threshold']
    w = params['weights']
    p_mods = params['primary_models']
    fb_mods = params['fallback_models']

    preds = []
    seen = set()
    for r in today_rows:
        name = r['name'].strip()
        if name in seen:
            continue
        seen.add(name)

        pc = num(r.get('pclose'))
        hi = num(r.get('phi'))
        lo = num(r.get('plo'))
        pv = num(r.get('pvolume'))
        mddv = num(r.get('mddv_20'))
        vix = num(r.get('vix'))
        vix_1m = num(r.get('vix_1m'))
        vix_3m = num(r.get('vix_3m'))
        vix_zs = num(r.get('vix_zscore_smooth'))

        hr14, hi14, shrunk_ratio = past_stats.get(name, (None, None, PRIOR_RATIO))
        u_row = user_today.get(name, {})
        log_indiavix = num(u_row.get('log_indiavix'))
        nsei_rv5d = num(u_row.get('nsei_rv5d'))

        vol_ratio = (pv / mddv) if (pv is not None and mddv is not None and mddv > 0) else None
        log_vol_ratio = math.log(max(vol_ratio, 1e-6)) if vol_ratio is not None else None
        log_mddv = math.log(max(mddv, 1.0)) if (mddv is not None and mddv > 0) else None

        if vix is not None and vix > 0 and pc is not None and pc > 0:
            daily_vol = vix / 100.0 / SQRT_252
            prev_range = (hi - lo) / pc if (hi is not None and lo is not None and hi >= lo) else None
            prev_range_filled = prev_range if prev_range is not None else hr14

            x_pri = [
                math.log(max(daily_vol, 1e-5)),
                vix_zs,
                (vix / vix_1m) if (vix_1m is not None and vix_1m > 0) else None,
                (vix / vix_3m) if (vix_3m is not None and vix_3m > 0) else None,
                math.log(max(prev_range_filled, 1e-5)) if prev_range_filled is not None else None,
                hr14,
                shrunk_ratio,
                log_vol_ratio,
                log_indiavix,
                nsei_rv5d,
                (vix / mkt_vix_mean) if (mkt_vix_mean is not None and mkt_vix_mean > 0) else None,
            ]
            p_huber = norm_cdf((score_linear_model(x_pri, p_mods['huber'], False) - LOG_2) / scale)
            p_ridge = norm_cdf((score_linear_model(x_pri, p_mods['ridge'], False) - LOG_2) / scale)
            p_logreg = score_linear_model(x_pri, p_mods['logreg'], True)
            p_dv = norm_cdf((daily_vol - params['dv_anchor_mu']) / params['dv_anchor_sd'])
            prob = w['huber'] * p_huber + w['ridge'] * p_ridge + w['logreg'] * p_logreg + w['dv'] * p_dv
        else:
            # Fallback path when vix/pclose are missing (e.g., masked practice dates or missing-VIX rows)
            x_fb = [
                math.log(max(hi14, 1e-5)) if hi14 is not None else None,
                hr14,
                shrunk_ratio,
                log_vol_ratio,
                log_mddv,
                log_indiavix,
                nsei_rv5d,
            ]
            p_fb_huber = norm_cdf((score_linear_model(x_fb, fb_mods['huber'], False) - LOG_2) / scale)
            p_fb_logreg = score_linear_model(x_fb, fb_mods['logreg'], True)
            prob = 0.5 * p_fb_huber + 0.5 * p_fb_logreg

        preds.append({'name': name, 'prediction': bool(prob >= thr)})

    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, a.date + '.predictions.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'prediction'])
        writer.writeheader()
        writer.writerows(preds)


if __name__ == '__main__':
    main()
