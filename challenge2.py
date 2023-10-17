import pandas as pd

from sklearn.metrics import f1_score
from pulsar_metrics.metrics.drift import CustomDriftMetric


@CustomDriftMetric
def test_custom(current, reference, multiple=3, **kwargs):
    # Calculate F1-score of actual and prediction values
    F1 = f1_score(current, reference)

    return F1


def test_custom(current, reference, multiple=3, **kwargs):
    """Return the F1 score of a machine learning model."""

    min_len = min(len(current), len(reference))

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(min_len):
        target = reference[i]
        prediction = current[i]

        if target == prediction == 1:
            tp += 1
        elif target == prediction == 0:
            tn += 1
        elif target == 0 and prediction == 1:
            fp += 1
        elif target == 1 and prediction == 0:
            fn += 1

    print("Accuracy analysis:")
    print("-------------------------")

    print("True Positives:", tp)
    print("True Negatives:", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)

    accuracy = (tn + tp) / (tn + tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 2 * (precision * recall) / (precision + recall)

    print("Accuracy: " + str(accuracy * 100) + "%")
    print("Precision: " + str(precision * 100) + "%")
    print("Recall: " + str(recall * 100) + "%")
    print("F1 Score: " + str(F1 * 100) + "%")
    print()

    return F1
