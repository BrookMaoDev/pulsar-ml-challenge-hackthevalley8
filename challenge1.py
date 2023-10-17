import numpy as np

from pulsar_metrics.metrics.drift import CustomDriftMetric

@CustomDriftMetric
def test_custom(current, reference, multiple=3, **kwargs):
    #Calculate z-scores of the feature for both the datasets

    std_dev_ref = np.std(reference)
    mean_ref = np.mean(reference)
    std_dev_cur = np.std(current)
    mean_cur = np.mean(current)

    z_score_ref = [(value - mean_ref) / std_dev_ref for value in reference]
    z_score_cur = [(value - mean_cur) / std_dev_cur for value in current]

    max_z_score_ref = max(z_score_ref)
    min_z_score_ref = min(z_score_ref)
    max_z_score_cur = max(z_score_cur)
    min_z_score_cur = min(z_score_cur)

    # return z-score_high - z-score_lows

