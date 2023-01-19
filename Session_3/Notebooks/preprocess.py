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
"""Feature engineer the customer churn dataset."""
# Python Built-Ins:
import argparse
import glob
import logging
import os
import pathlib

# External Dependencies:
import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_job_args():
    """Load job arguments from command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data",
        type=str,
        default="/opt/ml/processing/input/raw",
        help="(Local) folder location of the raw input data to be processed",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default="/opt/ml/processing/train",
        help="(Local) folder location to save train split of data",
    )
    parser.add_argument(
        "--output-validation",
        type=str,
        default="/opt/ml/processing/validation",
        help="(Local) folder location to save validation split of data",
    )
    parser.add_argument(
        "--output-test",
        type=str,
        default="/opt/ml/processing/test",
        help="(Local) folder location to save test split of data",
    )
    return parser.parse_args()


def read_csvs_recursive(folder: str) -> pd.DataFrame:
    """Read all .csv files in a folder to a combined DataFrame"""
    csv_files = glob.glob(folder + "/*.csv")
    chunks = []
    for filename in csv_files:
        chunks.append(pd.read_csv(filename, index_col=None))
    return pd.concat(chunks, axis=0, ignore_index=True)


if __name__ == "__main__":
    logger.info("Loading job configuration...")
    args = parse_job_args()
    logger.info(args)

    logger.info("Reading raw data...")
    df = read_csvs_recursive(args.input_data)

    logger.info("Processing...")

    # drop the "Phone" feature column
    df = df.drop(["Phone"], axis=1)

    # Change the data type of "Area Code"
    df["Area Code"] = df["Area Code"].astype(object)

    # Drop several other columns
    df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    # Convert categorical variables into dummy/indicator variables.
    model_data = pd.get_dummies(df)

    # Create one binary classification target column
    model_data = pd.concat(
        [
            model_data["Churn?_True."],
            model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
        ],
        axis=1,
    )

    logger.info("Splitting dataset...")
    # Split the data
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )

    logger.info("Writing output files...")
    os.makedirs(args.output_train, exist_ok=True)
    pd.DataFrame(train_data).to_csv(
        os.path.join(args.output_train, "train.csv"), header=False, index=False,
    )
    os.makedirs(args.output_validation, exist_ok=True)
    pd.DataFrame(validation_data).to_csv(
        os.path.join(args.output_validation, "validation.csv"), header=False, index=False,
    )
    os.makedirs(args.output_test, exist_ok=True)
    pd.DataFrame(test_data).to_csv(
        os.path.join(args.output_test, "test.csv"), header=False, index=False,
    )
