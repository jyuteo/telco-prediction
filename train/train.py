import os
import time
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RandomizedSearchCV

from utils import reset_seed, save_model
from models import ModelGenerator


def to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan


def preprocess_data(df):
    df.columns = list(map(lambda x: x[0].upper() + x[1:], df.columns))
    df = df.drop(columns=["CustomerID"])

    df["TotalCharges"] = df["TotalCharges"].apply(to_float)
    df["MonthlyOverTotalCharges"] = df["MonthlyCharges"] / df["TotalCharges"]

    target = "Default"
    numerical_features = sorted([
        "Tenure", "MonthlyCharges", "TotalCharges", "MonthlyOverTotalCharges"
    ])
    categorical_features = sorted(
        list(set(df.columns) - set(numerical_features) - {target}))

    df[numerical_features] = df[numerical_features].astype(float)
    df[categorical_features] = df[categorical_features].astype("category")

    df = df.dropna()
    X = df[numerical_features + categorical_features]
    y = (df[target] == "Yes").astype(int)
    return X, y, numerical_features, categorical_features


def load_data(path="./data/raw.csv"):
    data = pd.read_csv(path, index_col=False)
    return preprocess_data(data)


def get_row(model_name, result, time):
    mean_accuracy = np.nanmean(result["test_accuracy"])
    mean_recall = np.nanmean(result["test_recall"])
    mean_precision = np.nanmean(result["test_precision"])
    mean_f1 = np.nanmean(result["test_f1"])
    row = {
        "model": model_name,
        "mean_accuracy": mean_accuracy,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "time": time
    }
    return row


def train_model(model, X, y, model_name):
    start = time.time()
    reset_seed()
    if isinstance(model, RandomizedSearchCV):
        model.fit(X, y)
        params = model.best_params_
        model = model.estimator
        model.set_params(**params)

    reset_seed()
    result = cross_validate(estimator=model,
                            X=X,
                            y=y,
                            scoring=["accuracy", "precision", "recall", "f1"],
                            cv=5,
                            n_jobs=5)
    end = time.time()
    return model, get_row(model_name, result, end - start)


def save_results(result_ls, output_dir):
    results_df = pd.DataFrame(result_ls)
    results_df = results_df.sort_values(by='mean_f1',
                                        ascending=False,
                                        ignore_index=True)
    results_df.to_csv(os.path.join(output_dir, "cross_validation_results.csv"),
                      index=False)
    return results_df


def main(args):
    csv_path = args.input
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X, y, numerical_features, categorical_features = load_data(csv_path)
    all_results = list()
    all_models = dict()
    model_gen = ModelGenerator()

    for model_name, model in model_gen.generate_models(numerical_features,
                                                       categorical_features,
                                                       True):
        try:
            model, result = train_model(model, X, y, model_name)
            all_results.append(result)
            all_models[result["model"]] = model
            print(result)
            print("{} done, time taken {}".format(model_name, result["time"]))
            results_df = save_results(all_results, output_dir)
        except Exception as e:
            print("{} error: {}".format(model_name, e))

    model_name = results_df.loc[0, "model"]
    final_model = all_models[model_name]
    reset_seed()
    final_model.fit(X, y)
    save_model(final_model,
               os.path.join(output_dir, "{}.joblib".format(model_name)))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--input",
                        "-i",
                        type=str,
                        default="./data/raw.csv",
                        help="the path to the csv file")
    parser.add_argument("--output",
                        "-o",
                        type=str,
                        default="../model",
                        help="the directory to store training outputs")
    args = parser.parse_args()
    main(args)
