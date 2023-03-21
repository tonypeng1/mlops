import os
import sys
from pathlib import Path

import pandas as pd

import streamlit as st

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from config import config
from tagifai import main, utils

# from tagifai import main

# Title
st.title("MLOps Course · Made With ML")

# Sections
st.header("🔢 Data")
projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
df = pd.read_csv(projects_fp)
st.text(f"Projects (count: {len(df)})")
st.write(df)

st.header("📊 Performance")
performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Overall:")
st.write(performance["overall"])
tag = st.selectbox("Choose a tag: ", list(performance["class"].keys()))
st.write(performance["class"][tag])
tag = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
st.write(performance["slices"][tag])

st.header("🚀 Inference")
text = st.text_input(
    "Enter text:", "Adoption and effects of software engineering best practices in machine learning"
)
run_id = st.text_input("Enter run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
prediction = main.predict_tag(text=text, run_id=run_id)
st.write(prediction)
