
import numpy as np
from pulsar_metrics.metrics.drift import CustomDriftMetric

@CustomDriftMetric
def test_custom(current, reference, multiple=3, **kwargs):
    return multiple*np.max(current - reference)

@CustomDriftMetric
def test_custom(current, reference, multiple=3, **kwargs):
    #Calculate z-scores of the feature for both the datasets
    # return z-score_high - z-score_low

#Then we call our updated function as below
cus = test_custom(metric_name = 'test', feature_name = 'AveOccup') #it can be any numeric feature here
cus.evaluate(current = data_new, reference = data_ref, threshold = 1, multiple=0.5)

