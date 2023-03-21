"""# **Evaluation**

> Evaluating ML models by assessing overall, per-class and slice performances.
>
> * Compute the coarse-grained metrics such as overall precision, recall and
> f1 metrics.
>
> * Evaluate the same fine-grained metrics for each class (computer-vision, mlops, ...).
>
> * Inspect the metrics of key slices of our dataset that we need to do really well on:
>> * Features (explicit and implicit)
>>
>> * Metadata (timestamps, sources, etc.)
>>
>> * Priority slices / experience (minority groups, large customers, etc.)
>

"""

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from snorkel.slicing import PandasSFApplier, slicing_function

# from snorkel.slicing import slice_dataframe


@slicing_function()
def nlp_cnn(x):
    """NLP Projects that use convolution."""
    nlp_projects = "natural-language-processing" in x.tag
    convolution_projects = "CNN" in x.text or "convolution" in x.text
    return nlp_projects and convolution_projects


@slicing_function()
def short_text(x):
    """Projects with short titles and descriptions."""
    return len(x.text.split()) < 8  # less than 8 words


def get_slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, slices: np.recarray) -> Dict:
    """Generate metrics for slices of data.
    Args:
        y_true (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
        slices (np.recarray): generated slices.
    Returns:
        Dict: slice metrics.
    """
    metrics = {}
    for slice_name in slices.dtype.names:
        mask = slices[slice_name].astype(bool)
        if sum(mask):
            slice_metrics = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average="micro"
            )
            metrics[slice_name] = {}
            metrics[slice_name]["precision"] = slice_metrics[0]
            metrics[slice_name]["recall"] = slice_metrics[1]
            metrics[slice_name]["f1"] = slice_metrics[2]
            metrics[slice_name]["num_samples"] = len(y_true[mask])
    return metrics


def get_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: List, df: pd.DataFrame = None
) -> Dict:
    """Performance metrics using ground truths and predictions.
    Args:
        y_true (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
        classes (List): list of class labels.
        df (pd.DataFrame, optional): dataframe to generate slice metrics on. Defaults to None.
    Returns:
        Dict: performance metrics.
    """
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class metrics
    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    # Slice metrics
    # for i in [nlp_cnn, short_text]: print(slice_dataframe(df, i))
    if df is not None:
        slices = PandasSFApplier([nlp_cnn, short_text]).apply(df)
        metrics["slices"] = get_slice_metrics(y_true=y_true, y_pred=y_pred, slices=slices)

    return metrics
