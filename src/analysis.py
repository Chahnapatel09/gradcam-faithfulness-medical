import numpy as np
import pandas as pd


def load_results(path):
    df = pd.read_csv(path)
    df["group"] = np.select(
        [
            (df.true_label == 1) & (df.orig_pred == 1),
            (df.true_label == 0) & (df.orig_pred == 0),
            (df.true_label == 0) & (df.orig_pred == 1),
            (df.true_label == 1) & (df.orig_pred == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="unknown",
    )
    return df


def deletion_curves(df, group_cols=("cam_method", "percent", "strategy")):
    """Mean accuracy and confidence at each group, e.g. (cam_method, percent, strategy)."""
    return (
        df.groupby(list(group_cols))
        .agg(mean_accuracy=("masked_correct", "mean"),
             mean_confidence=("masked_conf", "mean"),
             n=("masked_correct", "size"))
        .reset_index()
    )


def faithfulness_gap(curves):
    """Pivot targeted vs random masking and compute the gap (random - targeted).

    A positive gap means targeted masking hurt the model more than random
    masking, i.e. the explanation is faithful. A gap near zero means the
    heatmap isn't pointing at pixels the model actually relies on.
    """
    pivot = curves.pivot_table(
        index=["cam_method", "percent"], columns="strategy",
        values=["mean_accuracy", "mean_confidence"],
    )
    pivot.columns = [f"{metric}_{strategy}" for metric, strategy in pivot.columns]
    pivot = pivot.reset_index()
    pivot["accuracy_gap"] = pivot["mean_accuracy_random"] - pivot["mean_accuracy_targeted"]
    pivot["confidence_gap"] = pivot["mean_confidence_random"] - pivot["mean_confidence_targeted"]
    return pivot


def deletion_curves_by_prediction_group(df):
    """Deletion curves split by TP / TN / FP (FN excluded — none exist in this dataset)."""
    return deletion_curves(df, group_cols=("group", "cam_method", "percent", "strategy"))


def add_confidence_bin(df):
    """Median-split each image's original prediction confidence into high/low."""
    per_image_conf = df.drop_duplicates("image_id")[["image_id", "orig_conf"]]
    median_conf = per_image_conf["orig_conf"].median()
    bin_map = per_image_conf.assign(
        confidence_bin=np.where(per_image_conf["orig_conf"] >= median_conf, "high", "low")
    ).set_index("image_id")["confidence_bin"]

    df = df.copy()
    df["confidence_bin"] = df["image_id"].map(bin_map)
    return df


def deletion_curves_by_confidence(df):
    """Deletion curves split by high/low original prediction confidence."""
    df = add_confidence_bin(df)
    return deletion_curves(df, group_cols=("confidence_bin", "cam_method", "percent", "strategy"))
