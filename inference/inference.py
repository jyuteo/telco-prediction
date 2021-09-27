import os
import json
import warnings
import argparse
import requests
import pandas as pd
import numpy as np


def to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def preprocess_data(df):
    df.columns = list(map(lambda x: x[0].upper() + x[1:], df.columns))
    df["TotalCharges"] = df["TotalCharges"].apply(to_float)
    df = df.dropna()
    return df


def load_data(path="./inference-data.csv"):
    data = pd.read_csv(path, index_col=False)
    return preprocess_data(data)


def get_predictions(api_url, request_body):
    timeout = 5
    try:
        response = requests.post(api_url, json=request_body, timeout=timeout)
        response = response.json()
    except:
        raise Exception("Cannot connect to {}".format(api_url))

    if "error" in response:
        raise Exception(response["error"])
    predictions = response["predictions"]
    return predictions


def main(args):
    inference_data_path = args.input
    inference_data_filename = os.path.basename(inference_data_path).split(
        ".")[0]

    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    api_url = args.api_url

    data = load_data(inference_data_path)
    data_json = json.loads("{\"data\":" + data.to_json(orient="records") + "}")

    customer_id_df = pd.DataFrame(data["CustomerID"])

    predictions = get_predictions(api_url, data_json)
    predictions_df = pd.DataFrame(predictions)
    predictions_df = pd.concat([customer_id_df, predictions_df], axis=1)
    predictions_df.columns = list(
        map(lambda x: x[0].upper() + x[1:], predictions_df.columns))
    predictions_df.to_csv(os.path.join(
        output_dir,
        "inference_results_{}.csv".format(inference_data_filename)),
                          index=False)
    print(predictions_df)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="inference.py")
    parser.add_argument("--input",
                        "-i",
                        type=str,
                        default="./data/data.csv",
                        help="the path to the csv file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./results",
        help="the directory to store csv file of predicted output")
    parser.add_argument("--api_url",
                        type=str,
                        default="http://localhost:5000/predict",
                        help="the url to the predict api endpoint")
    args = parser.parse_args()
    main(args)