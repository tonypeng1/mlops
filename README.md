### Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e ".[dev]"
pre-commit install
pre-commit autoupdate
```

> If the commands above do not work, please refer to the [packaging](https://madewithml.com/courses/mlops/packaging/) lesson. We highly recommend using [Python version](https://madewithml.com/courses/mlops/packaging/#python) `3.9.1`.

### Directory
```bash
tagifai/
├── data.py       - data processing components
├── evaluate.py   - evaluation components
├── main.py       - training/optimization operations
├── predict.py    - inference components
├── train.py      - training components
└── utils.py      - supplementary utilities
```

### Workflow
```bash
python tagifai/main.py elt-data
python tagifai/main.py optimize --args-fp="config/args.json" --study-name="optimization" --num-trials=10
python tagifai/main.py train-model --args-fp="config/args.json" --experiment-name="baselines" --run-name="sgd"
python tagifai/main.py predict-tag --text="Transfer learning with transformers for text classification."
```

### API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir tagifai --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```
