#!/bin/zsh

source venv/bin/activate
cd tests
great_expectations docs build
