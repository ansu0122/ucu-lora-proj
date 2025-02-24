#!/bin/bash

ENV_NAME=$(grep "name:" conda.yaml | awk '{print $2}')

source ~/miniconda3/bin/activate

if conda env list | grep -q "$ENV_NAME"; then
    echo "Removing existing Conda environment: $ENV_NAME..."
    conda env remove -n "$ENV_NAME"
fi

echo "Creating Conda environment from conda.yaml..."
conda env create -f conda.yaml

echo "Activating Conda environment: $ENV_NAME..."
source /opt/conda/bin/activate "$ENV_NAME"

echo "Done! Your Conda environment is installed."

