import pandas as pd


def test_custom(
    file: str, target_col_name: str, prediction_col_name: str
) -> float:
    """Return the F1 score of a machine learning model.
    'file' represents the CSV file
    'target_col_name' represents the column name of the correct answers
    'prediction_col_name' represents the column name of the model predictions
    """

    df = pd.read_csv(file)
    last_row_number = df.shape[0] - 1

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(last_row_number + 1):
        target = df.loc[i, target_col_name]
        prediction = df.loc[i, prediction_col_name]

        if target == prediction == 1:
            tp += 1
        elif target == prediction == 0:
            tn += 1
        elif target == 0 and prediction == 1:
            fp += 1
        elif target == 1 and prediction == 0:
            fn += 1

    print("Accuracy analysis using data from file", file)
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


test_custom("pulsar_metrics\data\california_ref.csv", "clf_target", "y_pred")
test_custom("pulsar_metrics\data\california_new.csv", "clf_target", "y_pred")
