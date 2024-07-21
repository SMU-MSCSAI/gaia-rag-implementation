#!/bin/bash

# Set the PYTHONPATH to the root of the project
export PYTHONPATH=$(pwd)

# Suppress Hugging Face tokenizers parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Run the script using the module's full import path
poetry run python -m gaia_framework.agents.agent_pipeline
