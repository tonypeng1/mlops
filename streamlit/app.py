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

# Title
st.title("A Language MLOps Project")

# Sections
st.header("🎸 Introduction")
st.markdown(
    """
This is the product of what I learned from a fantastic open source course on end-to-end
machine learning by Goku Mohandas. Name of the site is "Made With ML" ([site_url](https://madewithml.com/)) and the code repository
is at [repo_url](https://github.com/GokuMohandas/mlops-course/). This streamlit site is based
on my own repositoy ([site_url](https://github.com/tonypeng1/mlops)) with some added docstrings to explain the workflow
([mkdocs_url](https://tonypeng1.github.io/mlops/)).

This project aims to demonstrate an end-to-end MLOps process using a language model that
categorizes a ML paper to either computer vision, mlops, natural language processing, or others.
"""
)

st.header("🚀 Inference")
st.markdown(
    """
Want to see if this model correctly predicts the main topic of a machine learning paper? Please
input either the title of a paper or a combination of title and any sentences that describe the paper.
"""
)
text = st.text_input(
    "Enter text to predict the topic category of a paper:",
    "Deep Residual Learning for Image Recognition",
)
run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
prediction = main.predict_tag(text=text, run_id=run_id)
st.write(prediction)

st.header("🔢 Information: Data")
st.markdown(
    """
Raw data is listed in the table below, where 70% is used for training, 15% for validation,
and 15% for testing.
"""
)
projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
# os.system("pip install --force-reinstall -v 'fsspec==2022.11.0'")  # put fsspec==2022.11.0 into requirement.txt file
# os.system("pip show fsspec")
os.system("pip install dvc-gdrive")
os.system("dvc pull")
df = pd.read_csv(projects_fp)
st.text(f"All raw data (count: {len(df)})")
st.write(df)

st.header("📊 Information: Model Performance")
st.markdown(
    """
🍌 1. Overall performance of model is as follow.
"""
)
performance_fp = Path(config.CONFIG_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.write(performance["overall"])
st.markdown(
    """
🍒 2. Performace of model for each category is as follows.
"""
)
tag = st.selectbox("Choose a tag: ", list(performance["class"].keys()))
st.write(performance["class"][tag])
st.markdown(
    """
🍉 3. Finally, performace of model on sliced data is as follows. The first slice is NLP papers
that use convolution, and the second slice is papers with combined title and description
with less than 8 words.
"""
)
tag = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
st.write(performance["slices"][tag])
