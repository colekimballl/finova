#!/bin/bash

# Project Solaris AI - Cardano Trading System
# Conda Environment Setup Script

echo "🚀 Setting up Project Solaris AI Conda Environment..."

# Define environment name
ENV_NAME="cardano"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "⚠️ Environment '$ENV_NAME' already exists. Do you want to remove it and create a fresh one? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        conda env remove -n "$ENV_NAME"
    else
        echo "🛑 Setup aborted. Using existing environment."
        conda activate "$ENV_NAME"
        exit 0
    fi
fi

# Create a new conda environment
echo "📦 Creating new conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.9 -y

# Activate the environment
echo "🔄 Activating environment..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install required packages
echo "📚 Installing required packages..."
conda install -n "$ENV_NAME" pandas numpy matplotlib python-dotenv requests -y
conda install -n "$ENV_NAME" -c conda-forge ccxt -y

# Install pandas-ta and other packages using pip
echo "🔧 Installing additional packages with pip..."
pip install pandas-ta

# Create project directories if they don't exist
mkdir -p data/cardano
mkdir -p logs
mkdir -p strategies
mkdir -p backtest

echo "✅ Conda environment setup complete!"
echo ""
echo "📋 To activate this environment, use:"
echo "conda activate $ENV_NAME"
echo ""
echo "🚀 Project Solaris AI is ready for liftoff!"
