import pandas as pd
import numpy as np

def test_custom(current, reference, multiple=3, **kwargs):
    """Return the accuracy of a machine learning model."""

    df = pd.read_csv("pulsar_metrics\data\california_ref.csv")
    last_row_number = df.shape[0] - 1

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(last_row_number + 1):
        target = df.loc[i, "clf_target"]
        prediction = df.loc[i, "y_pred"]

        if target == prediction == 1:
            tp += 1
        elif target == prediction == 0:
            tn += 1
        elif target == 0 and prediction == 1:
            fp += 1
        elif target == 1 and prediction == 0:
            fn += 1

    print("True Positives:", tp)
    print("True Negatives:", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)

    print()

    accuracy = (tn + tp) / (tn + tp + fp + fn)
    print("Accuracy:", accuracy * 100, "%")


test_custom(1, 2, 3)