#!/bin/zsh

source venv/bin/activate
export AIRFLOW_HOME=${PWD}/airflow
export GOOGLE_APPLICATION_CREDENTIALS=/Users/tony3/Downloads/made-with-ml-384201-862d20d17018.json  # REPLACE
airflow webserver --port 8080
