#!/bin/bash

# Run the tests
# cd ./gaia_framework/

# Set the PYTHONPATH to the root of the project
export PYTHONPATH=$(pwd)

# Suppress Hugging Face tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false

poetry run python -m unittest discover -s ./gaia_framework/tests
