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

    z_scores_ref = zscore(metric_ref)
    z_scores_cur = zscore(metric_cur)
    
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
