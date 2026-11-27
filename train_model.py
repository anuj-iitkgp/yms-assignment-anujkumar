#!/usr/bin/env python3
"""Train the primary and fallback models on training data ONLY and export model_params.json.

Run after prepare_user_historical_info.py:
  python train_model.py
"""
import glob
import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, HuberRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PRIOR_RATIO = 1.28
PRIOR_WEIGHT = 8.0


def pipe_to_dict(pipe, is_clf=False):
    imp = pipe.named_steps['imp']
    sc = pipe.named_steps['sc']
    est = pipe.named_steps['clf' if is_clf else 'reg']
    coef = est.coef_[0].tolist() if is_clf else est.coef_.tolist()
    inter = float(est.intercept_[0]) if is_clf else float(est.intercept_)
    return {
        'medians': imp.statistics_.tolist(),
        'means': sc.mean_.tolist(),
        'scales': sc.scale_.tolist(),
        'coef': coef,
        'intercept': inter,
    }


def main():
    labels = pd.read_csv('data/train_labels.csv', dtype={'date': str})
    labels['moved'] = labels['moved'].astype(str).str.lower().isin(['true', '1', 'yes'])
    info_files = sorted(glob.glob('data/historical_info/*_n50.info.csv'))
    info = pd.concat([pd.read_csv(p, dtype={'date': str, 'vix_asof': str}) for p in info_files], ignore_index=True)
    user_info = pd.read_csv('user_historical_info.csv', dtype={'date': str})

    df = labels.merge(info, on=['date', 'name'], how='left')
    df = df.merge(user_info[['date', 'name', 'log_indiavix', 'nsei_rv5d']], on=['date', 'name'], how='left')
    df = df.sort_values(['name', 'date']).reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['log_move'] = np.log(df['max_move_pct'].clip(lower=0.05))

    df['daily_vol'] = df['vix'] / 100.0 / np.sqrt(252.0)
    df['prev_range'] = (df['phi'] - df['plo']) / df['pclose']
    df['prev_range_over_iv'] = df['prev_range'] / df['daily_vol']
    df['vol_ratio'] = df['pvolume'] / df['mddv_20']
    df['log_vol_ratio'] = np.log(df['vol_ratio'].clip(lower=1e-6))
    df['log_mddv'] = np.log(df['mddv_20'].clip(lower=1.0))
    df['vix_vs_1m'] = df['vix'] / df['vix_1m']
    df['vix_vs_3m'] = df['vix'] / df['vix_3m']

    df['mkt_vix_mean'] = df.groupby('date')['vix'].transform('mean')
    df['vix_rel_mkt'] = df['vix'] / df['mkt_vix_mean']

    df = df.sort_values(['name', 'dt']).reset_index(drop=True)
    hist_range_14d, hist_iv_14d, shrunk_ratio = [], [], []
    for _, grp in df.groupby('name', sort=False):
        dts = grp['dt'].values
        rngs = grp['prev_range'].values
        ivs = grp['daily_vol'].values
        ratios = grp['prev_range_over_iv'].values
        for i in range(len(grp)):
            if i == 0:
                hist_range_14d.append(np.nan)
                hist_iv_14d.append(np.nan)
                shrunk_ratio.append(PRIOR_RATIO)
            else:
                days_ago = (dts[i] - dts[:i]) / np.timedelta64(1, 'D')
                mask14 = days_ago <= 14.0
                r14 = rngs[:i][mask14]
                i14 = ivs[:i][mask14]
                hr = float(np.nanmean(r14)) if np.any(~np.isnan(r14)) else (float(np.nanmean(rngs[:i])) if np.any(~np.isnan(rngs[:i])) else np.nan)
                hi = float(np.nanmean(i14)) if np.any(~np.isnan(i14)) else (float(np.nanmean(ivs[:i])) if np.any(~np.isnan(ivs[:i])) else np.nan)
                hist_range_14d.append(hr)
                hist_iv_14d.append(hi)
                past_r = ratios[:i][~np.isnan(ratios[:i])]
                shrunk_ratio.append(float((np.sum(past_r) + PRIOR_WEIGHT * PRIOR_RATIO) / (len(past_r) + PRIOR_WEIGHT)))

    df['hist_range_14d'] = hist_range_14d
    df['hist_iv_14d'] = hist_iv_14d
    df['shrunk_ratio'] = shrunk_ratio
    df['prev_range_filled'] = df['prev_range'].fillna(df['hist_range_14d'])
    df['log_daily_vol'] = np.log(df['daily_vol'].clip(lower=1e-5))
    df['log_prev_range'] = np.log(df['prev_range_filled'].clip(lower=1e-5))
    df['log_hist_iv_14d'] = np.log(df['hist_iv_14d'].clip(lower=1e-5))

    u = df[df['pclose'].notna() & df['vix'].notna()].copy()
    f_primary = [
        'log_daily_vol', 'vix_zscore_smooth', 'vix_vs_1m', 'vix_vs_3m',
        'log_prev_range', 'hist_range_14d', 'shrunk_ratio',
        'log_vol_ratio', 'log_indiavix', 'nsei_rv5d', 'vix_rel_mkt',
    ]
    huber_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('reg', HuberRegressor(alpha=80.0, max_iter=500))]).fit(u[f_primary], u['log_move'].values)
    ridge_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('reg', Ridge(alpha=150.0))]).fit(u[f_primary], u['log_move'].values)
    logreg_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('clf', LogisticRegression(C=0.02, max_iter=1000))]).fit(u[f_primary], u['moved'].astype(int).values)

    f_fallback = [
        'log_hist_iv_14d', 'hist_range_14d', 'shrunk_ratio',
        'log_vol_ratio', 'log_mddv', 'log_indiavix', 'nsei_rv5d',
    ]
    fb_huber = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('reg', HuberRegressor(alpha=80.0, max_iter=500))]).fit(df[f_fallback], df['log_move'].values)
    fb_logreg = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler()), ('clf', LogisticRegression(C=0.02, max_iter=1000))]).fit(df[f_fallback], df['moved'].astype(int).values)

    params = {
        'threshold': 0.640,
        'weights': {'huber': 0.45, 'ridge': 0.20, 'logreg': 0.20, 'dv': 0.15},
        'primary_features': f_primary,
        'primary_models': {
            'huber': pipe_to_dict(huber_pipe, is_clf=False),
            'ridge': pipe_to_dict(ridge_pipe, is_clf=False),
            'logreg': pipe_to_dict(logreg_pipe, is_clf=True),
        },
        'fallback_features': f_fallback,
        'fallback_models': {
            'huber': pipe_to_dict(fb_huber, is_clf=False),
            'logreg': pipe_to_dict(fb_logreg, is_clf=True),
        },
        'log_move_scale': 0.48,
        'dv_anchor_mu': 0.017,
        'dv_anchor_sd': 0.004,
    }
    with open('model_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Trained primary on {len(u)} rows and fallback on {len(df)} rows -> model_params.json")


if __name__ == '__main__':
    main()
