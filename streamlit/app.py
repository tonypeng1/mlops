import os
import subprocess
import sys

# import tempfile
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
This is the product of a fantastic open-source course on end-to-end
machine learning by Goku Mohandas. Name of the site is "Made With ML" ([site_url](https://madewithml.com/)) and the code repository
is at [repo_url](https://github.com/GokuMohandas/mlops-course/).

This streamlit site is based
on my own repository ([site_url](https://github.com/tonypeng1/mlops)). The modification includes:

1. Added docstrings to explain the workflow ([mkdocs_url](https://tonypeng1.github.io/mlops/)),
2. DVC (data version control) data extraction from a Google service account.

This project aims to demonstrate an end-to-end MLOps process using a language model that
categorizes an ML paper to either computer vision, mlops, natural language processing, or others.
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
and 15% for testing. Data in the table below is pulled real-time from Google Drive.
"""
)
projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")

venv_path = sys.executable

def pull_data_with_dvc():
    cmd = ["which", "dvc"]
    # cmd = [venv_path, "-m", "dvc", "pull"]
    # result = subprocess.run(cmd, capture_output=True, text=True)
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = result.communicate()

    if result.returncode == 0:
        st.write("(🎉 Data pulled successfully!)")
        for line in stdout.decode().split("\n"):
            st.write(line)
    else:
        st.write("Error pulling data from Google Drive!")
        for line in stderr.decode().split("\n"):
            st.write(line)

    #     # st.write(result.stdout)
    # else:
    #     st.write("Error pulling data from Google Drive!")
    #     st.write(result.stderr)

# Use this function somewhere in your Streamlit app.
pull_data_with_dvc()

# # Create a temporary file
# with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as temp:
#     # Write a simple bash command to the file
#     temp.write("#!/bin/bash\n")
#     # temp.write("ls -a\n")
#     temp.write("which python\n")
#     # temp.write("which streamlit\n")
#     # temp.write("which mlflow\n")
#     # temp.write("which dvc\n")
#     # temp.write("python -m pip show dvc\n")
#     temp.write("echo $PATH\n")
#     temp.write("echo $SHELL\n")
#     temp.write("echo $VIRTUAL_ENV\n")
#     temp.write("echo $USER\n")
#     temp.write("hostname\n")
#     temp.write("pwd\n")
#     temp.write("ls -al /usr/local/bin\n")
#     temp.write("ls -al /usr/local/bin/python\n")
#     # temp.write("dvc doctor\n")
#     temp.write("/usr/local/bin/dvc doctor\n")
#     # temp.write("dvc pull\n")
#     temp_filename = temp.name

# # Make the temporary file executable
# os.chmod(temp_filename, 0o755)

# # Execute the script
# result = subprocess.run([temp_filename], capture_output=True, text=True)

# # Display the output
# st.write(result.stdout)

# result = subprocess.Popen(["bash", temp_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# stdout, stderr = result.communicate()

# st.write("Output:")
# for line in stdout.decode().split("\n"):
#     st.write(line)

# st.write("Error:")
# for line in stderr.decode().split("\n"):
#     st.write(line)

# st.write("Output:", stdout.decode())
# st.write("Error:", stderr.decode())

# # Remove the temp file
# os.remove(temp_filename)

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
