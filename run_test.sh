#!/bin/bash

# Run the tests
# cd ./gaia_framework/

# Set the PYTHONPATH to the root of the project
export PYTHONPATH=$(pwd)

poetry run python -m unittest discover -s ./gaia_framework/tests
