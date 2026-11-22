#!/usr/bin/env python3
"""Score <date>.predictions.csv files against truth labels (standard library only).

  test_predictions.py PREDICTIONS_DIR [--labels LABELS_CSV] [--min-precision 0.60] [--json OUT]

Each predictions file has columns name,prediction (True/False, 1/0, yes/no). Every labelled
(date, name) is scored; a missing file or row counts as a wrong prediction.
  precision = true positives / predicted moves      = TP / (TP + FP)
  recall    = true positives / actual moves (> 2%)  = TP / (TP + FN)
  combined  = 0.75 * precision + 0.25 * recall
  PASS      = precision > min-precision (default 0.60), plus optional gates recall >= min-recall
              and combined >= min-combined (both default 0 = off)
  (accuracy is reported for information only)
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRUE, FALSE = {'true', '1', 'yes', 't', 'y'}, {'false', '0', 'no', 'f', 'n'}
W_PREC, W_REC = 0.75, 0.25


def parse_bool(x):
    s = str(x).strip().lower()
    if s in TRUE:
        return True
    if s in FALSE:
        return False
    raise ValueError('not a boolean: %r' % x)


def load_labels(path):
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            out[(r['date'], r['name'])] = parse_bool(r['moved'])
    return out


def load_predictions(pred_dir, dates):
    preds, problems = {}, []
    for d in dates:
        p = os.path.join(pred_dir, d + '.predictions.csv')
        if not os.path.exists(p):
            problems.append('%s: missing file' % d)
            continue
        with open(p, newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {'name', 'prediction'} <= set(reader.fieldnames):
                problems.append('%s: needs columns name,prediction (got %s)' % (d, reader.fieldnames))
                continue
            for r in reader:
                try:
                    preds[(d, r['name'].strip())] = parse_bool(r['prediction'])
                except ValueError as e:
                    problems.append('%s %s: %s' % (d, r['name'], e))
    return preds, problems


def score(labels, preds):
    tp = fp = tn = fn = missing = 0
    for key, truth in labels.items():
        p = preds.get(key)
        if p is None:
            missing += 1
            p = not truth  # a missing prediction is always wrong
        if truth and p:
            tp += 1
        elif truth:
            fn += 1
        elif p:
            fp += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(rows=n, tp=tp, fp=fp, tn=tn, fn=fn, missing=missing, accuracy=acc, recall=rec,
                precision=prec, combined=W_PREC * prec + W_REC * rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('predictions_dir')
    ap.add_argument('--labels', default=os.path.join(HERE, 'data', 'test_labels.csv'))
    ap.add_argument('--min-precision', type=float, default=0.60)
    ap.add_argument('--min-combined', type=float, default=0.0, help='optional gate on the combined score')
    ap.add_argument('--min-recall', type=float, default=0.0, help='optional gate, e.g. 0.30, against predicting only a few certain moves')
    ap.add_argument('--json')
    a = ap.parse_args()
    labels = load_labels(a.labels)
    dates = sorted({d for d, _ in labels})
    preds, problems = load_predictions(a.predictions_dir, dates)
    s = score(labels, preds)
    s.update(dates=len(dates), first_date=dates[0], last_date=dates[-1], problems=problems,
             min_precision=a.min_precision, min_combined=a.min_combined, min_recall=a.min_recall,
             passed=s['precision'] > a.min_precision and s['combined'] >= a.min_combined and s['recall'] >= a.min_recall)
    print('dates %d (%s..%s)  rows %d  missing %d' % (s['dates'], s['first_date'], s['last_date'], s['rows'], s['missing']))
    print('confusion  TP %d  FP %d  TN %d  FN %d' % (s['tp'], s['fp'], s['tn'], s['fn']))
    print('precision %.4f\nrecall    %.4f\naccuracy  %.4f  (information only)' % (s['precision'], s['recall'], s['accuracy']))
    print('combined  %.4f   (= %.2f*precision + %.2f*recall)' % (s['combined'], W_PREC, W_REC))
    gate = 'precision > %.2f' % a.min_precision + (' and recall >= %.2f' % a.min_recall if a.min_recall else '') + (' and combined >= %.2f' % a.min_combined if a.min_combined else '')
    print('RESULT    %s   (needs %s)' % ('PASS' if s['passed'] else 'FAIL', gate))
    for p in problems[:20]:
        print('  problem:', p)
    if len(problems) > 20:
        print('  ... %d more problems' % (len(problems) - 20))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(s, f, indent=1)
    return 0 if s['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
