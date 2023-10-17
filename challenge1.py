import pandas as pd
import numpy as np

from scipy.stats import zscore
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
    print(z_score_ref)
    print(z_score_cur)
    # return z-score_high - z-score_lows

#def test_custom(current, reference, multiple=3, **kwargs):
    #Calculate z-scores of the feature for both the datasets

    #z_scores_cur = zscore(current)
    #z_scores_ref = zscore(reference)
    
    # return z-score_high - z-score_lows

def custom_metric(file, avg_rooms, avg_occupancy):
    """Return the rooms to occupany ratio.
    'file' represents the CSV file
    'avg_rooms' represents the column name of average number of rooms
    'avg_occupancy' represents the column name of average number of people per household
    """

    df = pd.read_csv(file)
    last_row_number = df.shape[0] - 1

    total_rooms = 0
    total_occupants = 0

    for i in range(last_row_number):
        rooms = df.loc[i, avg_rooms]
        occupancy = df.loc[i, avg_occupancy]

        total_rooms += rooms
        total_occupants += occupancy

    return total_rooms / total_occupants
