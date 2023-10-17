import pandas as pd
import numpy as np

from pulsar_metrics.metrics.drift import CustomDriftMetric

@CustomDriftMetric
def test_custom(current, reference, multiple=3, **kwargs):
    #Calculate z-scores of the feature for both the datasets
    ref = pd.read_csv(reference)
    cur = pd.read_csv(current)

    metric_ref = ref[kwargs]
    metric_cur = cur[kwargs]

    std_dev_ref = np.std(metric_ref)
    std_dev_cur = np.std(metric_cur)
    mean_ref = np.mean(metric_ref)
    mean_cur = np.mean(metric_cur)

    z_score_ref = [(value - mean_ref) / std_dev_ref for value in metric_ref]
    z_score_cur = [(value - mean_cur) / std_dev_cur for value in metric_cur]

    max_z_score_ref = max(z_score_ref)
    min_z_score_ref = min(z_score_ref)
    max_z_score_cur = max(z_score_cur)
    min_z_score_cur = min(z_score_cur)

    # return z-score_high - z-score_lows

