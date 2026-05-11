# Retraining Policy

Retraining should be considered when one or more of these conditions is true:

- Production MAE or RMSE is materially worse than validation performance.
- Input feature distributions drift from the training reference window.
- Market regime changes create sustained forecast bias.
- New SMARD data covers a meaningful additional time period.
- The serving model is older than the agreed freshness window.

Initial policy:

- Recompute monitoring metrics weekly.
- Review drift and error trends monthly.
- Retrain manually until the pipeline is stable enough for scheduled retraining.

