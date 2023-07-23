"""
# **Model Training**

## 1. 🏝️  **In `train` function:**

> 1. Set random seed so that the same sequence of random numbers is generated every time
> `random.random()` or `np.random.rand()` is run.
>
> 2. Use `df.sample(fract=1).reset_index(drop=True)` (randomly sample rows from a Panda dataframe)
> to shuffle 100% of the rows of `df` (do not keep the old index).
>
> 3. (in `data.preprocess()`) First preprocess the `df`:
>> 1. Feature engineering: Merge column `title` and `description` to form a new column `text`.
>>
>> 2. Clean text:
>>       * Change to lower case
>>       * Remove stopwords
>>       * Add spacing between objects to be filtered
>>       * Remove non alphanumeric chars (only use A-Za-z0-9)
>>       * Remove multiple spaces
>>       * Strip white space at the ends
>>       * Remove links
>>       * Stemming
>>
>> 3. Replace out of scope labels (not in the accepted label list) with "other".
>>
>> 4. Replace labels whose occurrences are less than a certain frequency with "other".
>>
> 4. Initiate an instance of `LabelEncoder` first, and use its `.fit()` methosd to fit each unique
> label to an index, and assign to instance attributes of `.class_to_index` and `.index_to_class`.
> Finally return the instance.
>
> 5. Change `text` column to a numpy array, encode the `tag` column using `.encode()`, and
> train-test split the `df` to 70% training data, 15% validation data, and 15% test data
> (`stratify = y`).
>
> 6. Construct `test` and decoded labels back to a dataframe of test data.
>
> 7. Initiate an instance of `TfidfVectorizer()` with the optimized `analyzer` = `char_wb`
> (the input text is tokenized into charater n-gram within word boundaries, which are defined as
> the edges of word tokens or the spaces between words) and the `ngram_range = (2, 6)`.
>
> 8. Use vectorizer to `.fit_transform()` `X_train` (resulting in 668x27096 sparse matrix of type
> `numpy.float64`) and to `vectorizer.fransform()` `X_val` and `X_test`.
>
> 9. From `imblearn.over_sampling` import `RandomOverSample()` to randomly sample the minority
> classes (with replacement) from `X_train` and `y_train` and add samples back to the class until
> all classes have the same number of samples (`X_over`, `y_over`).
>
> 10. Initiate a model instance of `SGDClassifier`. Set `max_iter` = 1. But iterate through ecpoches
> using `range(arg.num_epochs)` (= 100). Each time `model.fit(X_over, y_over)` is called the model
> iterate once.
>
> 11. Use `log_loss()` to calculate the cross-entropy loss between `y_train` and `model.predict_proba(X_train)`
> (= `train_loss`) and between `y_val` and `model.predict_proba(X_val)` for each epoch.
>
> 12. Use `mlflow.log_metrics({})` to log `train_loss` and `val_loss`.
>
> 13. After 100 iterations, use the final model to `.predict(X_val)` and `.predict_proba(X_val)`.
> Inspect the predicted label's probability of each sample, and find the 1st quartile probability
> (using `np.quantile(data, q=0.25)`).
>
> 14. Find the index of the `other` class (= 3), use the model to predict (`.predict_proba()`) `X_test`'s
> probability (e.g., ([0.05, 0.87, 0.05, 0.03])). If the predicted label's probability is smaller
> than the threshold, change the prediction to `3` (`others`), and save as the final prediction (= `y_pred`).
>
> 15. Use `precision_recall_fscore_support(y_true, y_pred, average="weighted")` to calculate metric values.
> Here `weighted` means to calculate metrics for each label, and find their average weighted by support
> (= the number of occurrences of each class in `y_true`).
>
> 16. Also use the same function with `average=None` to calculate the scores for each class.
>
> 17. Use `snorkel.slicing`'s `@slicing_function()` decorator to define two slicing functions:
> `nlp_cnn()` (NLP projects that use convolution) and `short_text()` (project with combined title and
> description with less than 8 words).
>
> 18. Then use `PandasSFApplier()` to apply these two slicing functions to `df`. This function returns
> numpy record array: `slices = rec.array([(0, 0) (not true for both), (1, 0) (true for nlo_cnn), ...,
> (0, 1) (true for short_text), ..., (0, 0)], dtype=[("nlp_cnn", "<i8"), ("short_text", "<i8")])`.
> `slices["nlp_cnn"]` lists all values in the first value of a tuple, and `slices["short_text"]` lists
> all values in the second value of a tuple.
>
> 19. Change `slices["short_text"]` (array([0, 0, ..., 1, 0 , ...])) to `.astype(bool)` to a mask, and
> apply it to `y_true` and `y_pred` (using `y_true[mask]` and `y_pred[mask]`), then use
> `precision_recall_fscore_support()` with `average = "micro"` to calculate metric scores (part of
> `performance`).
>
> 20. Finally, `train.train()` returns `args`, `label_encoder`, `vectorizer`, `model`, and `performance`
>

## 2. ⛩️  **Use `get_data_splits` to split data (in `data.get_data_split` module):**

> * Training split (e.g., 70%) to train the model
>> Here the model has access to both inputs and outputs to optimize its internal weights.
> * Validation split (e.g., 15%) to determine model performance after each training loop
> (epoch)
>> Model does not use the outputs to optimize its weights but instead, the output is used
>> to optimize training hyperparameters such as the `learning_rate`, `analyzer`, `ngram_max_range`, etc.
>> (see `main.py/optimize`).
> * Test split (e.g., 15%) to perform a one-time assessment of the model.
>> Our best measure of how the model may behave on new, unseen data.
"""

import json
import os
import sys
from argparse import Namespace

# from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config.config import logger
from tagifai import data, evaluate, predict, utils


def train(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial = None) -> Dict:
    """Train model on data.
    Args:
        args (Namespace): parameter arguments to use for training.
        df (pd.DataFrame): data for training.
        trial (optuna.trial._trial.Trial, optional): optimization trial. Defaults to None.
    Raises:
        optuna.TrialPruned: early stopping of trial if it's performing poorly.
    Returns:
        Dict: artifacts from the run.
    """
    # Setup
    utils.set_seeds()
    if args.shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = df[: args.subset]  # None = all samples
    df = data.preprocess(df, lower=args.lower, stem=args.stem, min_freq=args.min_freq)
    label_encoder = data.LabelEncoder().fit(df.tag)
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
        X=df.text.to_numpy(), y=label_encoder.encode(df.tag)
    )
    test_df = pd.DataFrame({"text": X_test, "tag": label_encoder.decode(y_test)})

    # Tf-idf
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer, ngram_range=(2, args.ngram_max_range)
    )  # char n-grams
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # Oversample
    oversample = RandomOverSampler(sampling_strategy="all")
    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # Model
    model = SGDClassifier(
        loss="log",
        penalty="l2",
        alpha=args.alpha,
        max_iter=1,
        learning_rate="constant",
        eta0=args.learning_rate,
        power_t=args.power_t,
        warm_start=True,
    )

    # Training
    for epoch in range(args.num_epochs):
        model.fit(X_over, y_over)
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))
        if not epoch % 10:
            print(
                f"Epoch: {epoch:02d} | "
                f"train_loss: {train_loss:.5f}, "
                f"val_loss: {val_loss:.5f}"
            )
        # Log
        if not trial:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # Pruning (for optimization in next section)
        if trial:  # pragma: no cover, optuna pruning
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Threshold
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)
    args.threshold = np.quantile([y_prob[i][j] for i, j in enumerate(y_pred)], q=0.25)  # Q1

    # Evaluation
    other_index = label_encoder.class_to_index["other"]
    y_prob = model.predict_proba(X_test)
    y_pred = predict.custom_predict(y_prob=y_prob, threshold=args.threshold, index=other_index)
    performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred, classes=label_encoder.classes, df=test_df
    )

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


def objective(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial) -> float:
    """Objective function for optimization trials.
    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
        trial (optuna.trial._trial.Trial, optional): optimization trial.
    Returns:
        float: metric value to be used for optimization.
    """
    # Parameters to tune
    args.analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    args.ngram_max_range = trial.suggest_int("ngram_max_range", 3, 10)
    args.learning_rate = trial.suggest_loguniform("learning_rate", 1e-2, 1e0)
    args.power_t = trial.suggest_uniform("power_t", 0.1, 0.5)

    # Train & evaluate
    artifacts = train(args=args, df=df, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]["overall"]
    logger.info(json.dumps(overall_performance, indent=2))
    trial.set_user_attr("precision", overall_performance["precision"])
    trial.set_user_attr("recall", overall_performance["recall"])
    trial.set_user_attr("f1", overall_performance["f1"])

    return overall_performance["f1"]
