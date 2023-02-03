# virtual environment
python3 -m venv venv
source venv/bin/activate
python3 -m pip install pip setuptools wheel
python3 -m pip install -e .
pre-commit install
pre-commit autoupdate

# work flow
python tagifai/main.py elt-data
python tagifai/main.py optimize --args-fp="config/args.json" --study-name="optimization" --num-trials=10
python tagifai/main.py train-model --args-fp="config/args.json" --experiment-name="baselines" --run-name="sgd"
python tagifai/main.py predict-tag --text="Transfer learning with transformers for text classification."

# api
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
