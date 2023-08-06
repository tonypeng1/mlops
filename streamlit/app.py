import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pipdeptree

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
and 15% for testing.
"""
)
projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
st.write(config.DATA_DIR)
# os.system("pip install --force-reinstall -v 'fsspec==2022.11.0'")  # put fsspec==2022.11.0 into requirement.txt file
# os.system("pip show fsspec")
# os.system("pip install dvc-gdrive")

# os.system("dvc pull")
# if os.system("dvc pull") != 0:
#     st.text("dvc pull failed")
# st.text(projects_fp)
# if os.path.isfile("/mount/src/mlops/data/labeled_projects.csv"):
#     # if os.path.isfile(projects_fp):
#     st.text("File exists.")
# else:
#     st.text("File not found.")
# os.system("pipdeptree --packages dvc --warn > dependency.txt")
# if os.system("pipdeptree --packages dvc --warn > dependency.txt") != 0:
#     st.text("pipdeptree failed")
# if os.path.isfile("/mount/src/mlops/dependency.txt"):
#     # if os.path.isfile(projects_fp):
#     st.text("dependency.txt file exists.")
# else:
#     st.text("dependency.txt file not found.")
# with open("dependency.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         st.text(line)

# # Create a temporary file
# temp = tempfile.NamedTemporaryFile(delete=False)

# # Run the shell command and redirect the output to the temporary file
# os.system("ls > " + temp.name)

# # Now you can read the output from the file
# with open(temp.name, 'r') as file:
#     output = file.read()

# print(output)

# # Don't forget to remove the temporary file when you're done with it
# os.unlink(temp.name)

# Create a temporary file
with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as temp:
    # Write a simple bash command to the file
    temp.write("#!/bin/bash\n")
    temp.write("ls data\n")
    temp.write("dvc pull\n")
    temp.write("ls data\n")
    temp.write("find . -name 'labeled_projects.csv'")
    temp_filename = temp.name

# Make the temporary file executable
os.chmod(temp_filename, 0o755)

# Execute the script
result = subprocess.run([temp_filename], capture_output=True, text=True)

# Display the output
st.write(result.stdout)

# # Split the output into lines
# lines = result.stdout.splitlines()

# # Print each line
# for line in lines:
#     st.text(line)

# with open("dependency.txt") as f:
#     lines = f.readlines()
#     for line in lines:
#         st.text(line)

# Clean up the temporary file
os.remove(temp_filename)


# # Path to the shell script
# script_path = "./streamlit/date_script.sh"

# # Run the shell script and capture the output
# result = subprocess.run([script_path], capture_output=True, text=True)

# # Display the output in the Streamlit app
# if result.stderr:
#     st.write("Error:", result.stderr)
# else:
#     st.write("Output:", result.stdout)

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
