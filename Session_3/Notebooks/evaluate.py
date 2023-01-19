# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Evaluation script for measuring model accuracy."""
# Python Built-Ins:
import json
import logging
import os
import pickle
import tarfile

# External Dependencies:
import pandas as pd
import xgboost

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# May need to import additional metrics depending on what you are measuring.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
from sklearn import metrics as skmetrics


if __name__ == "__main__":
    logger.debug("Loading xgboost model")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")

    model = xgboost.Booster()
    model.load_model("xgboost-model")
    # Use pickle option instead for xgb<1.3:
    # model = pickle.load(open("xgboost-model", "rb"))

    logger.info("Loading test input data")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data")
    predictions = model.predict(X_test)

    logger.info("Creating classification evaluation report")
    predicted_labels = predictions.round()  # (Assumed 0.5 threshold here)
    acc = skmetrics.accuracy_score(y_test, predicted_labels)
    auc = skmetrics.roc_auc_score(y_test, predicted_labels)
    f1 = skmetrics.f1_score(y_test, predicted_labels)

    # Confusion matrix calc:
    actual_true_mask = y_test == 1
    actual_true_preds = predicted_labels[actual_true_mask]
    n_actual_true_pred_true = int(actual_true_preds.sum())
    actual_false_preds = predicted_labels[~actual_true_mask]
    n_actual_false_pred_true = int(actual_false_preds.sum())
    fpr, tpr, _ = skmetrics.roc_curve(y_test, predictions)

    # The metrics reported can change based on the model used, but it must be a specific name per:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": auc, "standard_deviation": "NaN"},
            "confusion_matrix": {
                # Structure of confusion_matrix field is {actual}->{predicted}
                "0": {
                    # Being careful with types here because numpy.float32 is not JSON serializable
                    "0": int(len(actual_false_preds)) - n_actual_false_pred_true,
                    "1": n_actual_false_pred_true,
                },
                "1": {
                    "0": int(len(actual_true_preds)) - n_actual_true_pred_true,
                    "1": n_actual_true_pred_true,
                },
            },
            "f1": {"value": f1, "standard_deviation": "NaN"},
            "receiver_operating_characteristic_curve" : {
                "false_positive_rates": fpr.tolist(),
                "true_positive_rates": tpr.tolist(),
            },
        },
    }

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join(
        "/opt/ml/processing/evaluation", "evaluation.json"
    )
    logger.info("Saving classification report to %s", evaluation_output_path)
    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
