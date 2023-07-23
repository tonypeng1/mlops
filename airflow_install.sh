#!/bin/zsh

source venv/bin/activate
export AIRFLOW_HOME=${PWD}/airflow
AIRFLOW_VERSION=2.3.3
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

pip install google-cloud-bigquery==1.21.0
pip install airflow-provider-great-expectations==0.1.1
