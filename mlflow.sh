#!/bin/zsh

source venv/bin/activate
mlflow server -h 0.0.0.0 -p 4000 --backend-store-uri $PWD/stores/model
