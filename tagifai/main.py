"""# **Core Operations**

## 1. 🏝️  **In `optimize` function:**

###(To run via CLI: type `python tagifai/main.py optimize`)

> 1. Load the argumrnt json file into a Python dictionary, followed by unpacking it as keyword arguments
> and passing it to the `Namespace` constructor (from `argparse` library), which allows you to access
> the elements of the dictionary using dot notation.
>
> 2. Set `optuna` pruner to `MedianPruner`, which stops a trial if the trial's best intermediate result
> is worse than median of intermediate results of previous trials at the same step.
>
> 3. Initiate a `optuna` study using the default study_name of "optimization".
>
> 4. Initiate an instance of `MLflowCallback` (from `optuna.integration.mlflow`) to track relevant
> information of Optuna.
>
> 5. Start optimization using `study.optimize()` by calling `train.objective` for `n_trial` times with `trial`
> (= `optuna.trial._trial.Trial` type) and the current args as input.
>
> 6. (in `train.objective`) Set the parameters to tune using, e.g., `trial.suggest_loguniform(
> learning_rate, 1e-2, 1e0)` to suggest values for this continuous parameter.
>
> 7. Pass `df`, new args, and `trial` to `train.train()`.
>
> 8. (in `train.train()`) Use the new args to train, in each of the training epoch (from 1 to
> `args.num_epochs` = 100) uses `trail.report()` to report `val_loss` and `epoch`, which are used to
> determine whether this `trail` should be pruned.
>
> 9. Return artifacts (`args`, `label_encoder`, `vectorizer`, `model`, and `performance`).
>
> 10. (in `train.objective()`) Use, e.g., `trail.set_user_attr(precision, artifacts[performance]
> [overall][precision]`) to set user attributes to the trial.
>
> 11. Retrun `f1`.
>
> 12. (in `main.optimize()`), in `study.optimize()` creates (if not already exists) a new MLflow
> experiment with the name `optimization`.
>
> 13. Go to Step 6 and continue to the next trial.
>
> 14. After `num_trails` is completed, merge the dictionary of `study.best_trail.params` into the
> starting args `args.__dict__` using `args = {**args.__dict__, **study.best_trial.params}`. Overide
> the values if the starting args have keys of the same names.
>
> 15. Save the optimized args into the `config/args.json` file location.

## 2. ⛺  **In class `LabelEncoder`**

> 1. `fit` raw labels to get a class instance with attributes `class_to_index` and
> `index_to_class` (in `train.train` module).
>
> 2. Then `encode` raw labels for further training (in `train.train` module).
>
> 3. After traing is completed, `save` `class_to_index` to MLflow folder stores/model/1/
> 5457....0020/artifacts/label_encoder.json (in `main.train_model` module).

## 3. ⛵  **In `train_model`**

###(To run via CLI: type `python tagifai/main.py train-model`)

> 1. Load the same raw data `df`.
>
> 2. Load the optimized args.json file as a Namespace object.
>
> 3. Set the MLflow experiment name to `baseline` using `mlflow.set_experiment()`. (Tracking uri has
> already been set by `mlflow.set_tracking_uri()` in `config/config.py`)
>
> 4. Start a mlflow run with the run name = `run_name` (= `sgd`).
>
> 5. Get `run_id` using `mlflow.active_run().info.run_id`.
>
> 6. (in `train.train`) Input `df` and `args` to `train()`, where the `train_loss` and `val_loss` of
> each epoch are logged using `mlflow.log_metrics(Dict[str, float], Step: optional [int])`.
>
> 7. (in `main.train-model`) Get the artifacts dictionary back from `train.train()`, log additional
> metrics of overall precision, recall, and f1 using `mlflow.log_metrics()`.
>
> 8. Save the parameters using `mlflow.log_params(Dict[str, Any])` after converting the Namespace args
> (by returning the `__dict__` method of Namespace) to dictionary using `vars()`.
>
> 9. Use `mlflow.log_artifacts(local_dir: str)` to log all the contents of a local directory as artifacts
> of this run. Here we use `with tempfile.TemporaryDirectory() as dp:` to create a temporary directory to
> save the artifacts.
>> 1. Use `json.dump()` to save the dictionary of args to a json file using the serialization encoder
>> class l`NumpyEncoder`.
>>
>> 2. Use the `save()` method of a `LableEncoder` class instance to save the `class_to_index` (key) dictionary
>> to a json file.
>>
>> 3. Use `joblib.dump()` on the `vectorizer` (a `TfidfVectorizer` instance) and `model` (a `SGDClassifier`
>> instance) to save them as `.pkl` files.
>>
>> 4. Use `json.dump()` to save the performance dictionary to a json file.
>>
>> 5. Finally use `mlflow.log_artifacts(dp)` to log all the contents of the temporary directory `dp`.
>
> 10. If not a test,
>> 1. Use `Path(config.CONFIG_DIR, "run_id.txt")` to create a platform-independent file path and open it in
>> write mode. Then use the `write()` method of `open(Path(), "w")` to write `run_id` to `run_in.txt`.
>>
>> 2. Use `json.dump()` to save the performance dictionary to `performance.json`.

"""

import json
import os
import sys
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Dict

import joblib
import mlflow
import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# from distutils.util import strtobool

from config import config
from config.config import logger
from tagifai import data, predict, train, utils

warnings.filterwarnings("ignore")
# logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def elt_data():
    """Extract, load and transform our data assets."""
    # Extract + Load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()]  # drop rows w/ no tag
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

    # logger.info("✅ Saved data!")


@app.command()
def train_model(
    args_fp: str = "config/args.json",
    experiment_name: str = "baselines",
    run_name: str = "sgd",
    test_run: bool = False,
) -> None:
    """Train a model given arguments.
    Args:
        args_fp (str): location of args.
        experiment_name (str): name of experiment.
        run_name (str): name of specific run in experiment.
        test_run (bool, optional): If True, artifacts will not be saved. Defaults to False.
    """
    logger.info(f"\nargs_fp: {args_fp}")
    logger.info(f"experiment_name: {experiment_name}")
    logger.info(f"run_name: {run_name}")
    logger.info(f"test_run: {test_run}")

    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info(f"artifact_uri: {mlflow.get_artifact_uri()}")
        logger.info(f"tracking_uri: {mlflow.get_tracking_uri()}")
        logger.info(f"Run ID: {run_id}")
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        # performance = artifacts["performance"]
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            logger.info(f"artifact_uri: {mlflow.get_artifact_uri()}")
            logger.info(f"tracking_uri: {mlflow.get_tracking_uri()}")
            mlflow.log_artifacts(dp)

    # Save to config
    if not test_run:  # pragma: no cover, actual run
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def optimize(
    args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
) -> None:
    """Optimize hyperparameters.
    Args:
        args_fp (str): location of args.
        study_name (str): name of optimization study.
        num_trials (int): number of trials to run in study.
    """
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    logger.info(f"\nTracking_uri: {mlflow.get_tracking_uri()}")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    logger.info(f"\nargs.__dict__: {args.__dict__}")
    logger.info(f"study.best_trial.params: {study.best_trial.params}")
    args = {**args.__dict__, **study.best_trial.params}
    logger.info(f"arg: {args}")
    utils.save_dict(d=args, filepath=args_fp, cls=NumpyEncoder)
    logger.info(f"\nBest value (f1): {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


def load_artifacts(run_id: str = None) -> Dict:
    """Load artifacts for a given run_id.
    Args:
        run_id (str): id of run to load artifacts from.
    Returns:
        Dict: run's artifacts.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(fp=Path(artifacts_dir, "label_encoder.json"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


@app.command()
def predict_tag(text: str = "", run_id: str = None) -> None:
    """Predict tag for text.

    Args:
        text (str): input text to predict label for.
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    # args_fp = Path(config.CONFIG_DIR, "args.json")
    # train_model(args_fp)

    # from config import config
    # from tagifai import main
    # args_fp = Path(config.CONFIG_DIR, "args.json")
    # optimize(args_fp, study_name="optimization", num_trials=20)

    # args_fp = Path(config.CONFIG_DIR, "args.json")
    # train_model(args_fp, experiment_name="baselines", run_name="sgd")

    # text = "Transfer learning with transformers for text classification."
    # run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    # predict_tag(text=text, run_id=run_id)

    app()
