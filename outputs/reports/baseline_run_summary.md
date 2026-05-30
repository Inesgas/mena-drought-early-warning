# Baseline Forecast Run

- Train period ends: `2018-12-01`
- Validation period ends: `2021-12-01`
- Horizons: `1, 3, 6`
- Optional XGBoost requested: `no`

## Best Model by Horizon

- t+1: `Persistence` (macro F1 `0.744`, balanced accuracy `0.744`)
- t+3: `Persistence` (macro F1 `0.594`, balanced accuracy `0.593`)
- t+6: `Persistence` (macro F1 `0.398`, balanced accuracy `0.401`)

## Spatial Holdout Check

- t+1 Random Forest mean macro F1 across held-out regions: `0.561`
- t+3 Random Forest mean macro F1 across held-out regions: `0.509`
- t+6 Random Forest mean macro F1 across held-out regions: `0.433`

## Notes

- Metrics are computed on the held-out test period after the validation cutoff.
- Monthly climatologies are estimated from the training period by default.
- Persistence is the minimum benchmark to beat: it predicts future drought class from the issue-month class.
