from constants import *
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def evaluate_model(Method, phase):
    """
    1. load test input data (and dev data if phase=="dev")
    2. apply Method.run(...) on the data
    3. save predictions under ./output/<phase>_predictions.csv
    """
    X_train = None
    y_train = None
    if phase == "dev":
        with open("data/X_train.pickle/X_train.pickle", "rb") as f:
            X_train = pickle.load(f)
        with open("data/y_train.pickle/y_train.pickle", "rb") as f:
            y_train = pickle.load(f)

    with open("../scripts/test_data/X_test_reduced.pickle/X_test_reduced.pickle", "rb") as f:
        X_test = pickle.load(f)

    # run the method
    df_pred = Method.run(
        phase=phase,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model_directory_path="model_dir_NN",
        id_column_name="id",
        prediction_column_name="pred"
    )

    # save to output folder
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", f"{phase}_predictions.csv")
    df_pred.to_csv(output_path, index=False)

def get_score(Method, phase):
    """
    1. load predictions from ./output/<phase>_predictions.csv
    2. load true labels from y_test_reduced.pickle
    3. compute accuracy and balanced accuracy
    4. print metrics and return balanced accuracy
    """
    # load predictions
    pred_path = os.path.join("output", f"{phase}_predictions.csv")
    df_pred = pd.read_csv(pred_path)

    # load ground truth
    with open("../scripts/test_data/y_test_reduced.pickle/y_test_reduced.pickle", "rb") as f:
        y_test = pickle.load(f)

    # build true-label DataFrame
    records = []
    for name, adj in y_test.items():
        for parent in adj.index:
            for child in adj.columns:
                records.append({
                    "id": f"{name}_{parent}_{child}",
                    "true": int(adj.loc[parent, child])
                })
    df_true = pd.DataFrame(records)

    # merge and score
    df = pd.merge(df_true, df_pred, on="id")
    y_true = df["true"]
    y_pred = df["pred"]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    print("Overall accuracy:  ", acc)
    print("Balanced accuracy:", bal_acc)

    return bal_acc
