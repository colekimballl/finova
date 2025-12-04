#!/bin/bash

# Project Solaris - Fix Conda Environment
# This script installs missing packages in the conda environment

echo "🔧 Fixing Cardano Trading Bot conda environment"

# Determine if we're in the conda environment
if [[ $CONDA_DEFAULT_ENV != "cardano" ]]; then
    echo "⚠️ Please activate the conda environment first with: conda activate cardano"
    echo "   Then run this script again."
    exit 1
fi

echo "✅ Working in the 'cardano' conda environment"

# Install missing packages
echo "📦 Installing missing packages..."

# Install matplotlib with conda
echo "📊 Installing matplotlib..."
conda install -y matplotlib

# Install python-dotenv with pip
echo "🔐 Installing python-dotenv..."
pip install python-dotenv

# Install pandas-ta with pip
echo "📈 Installing pandas-ta..."
pip install pandas-ta

# Try to install ccxt with pip (optional)
echo "🌐 Installing ccxt (optional)..."
pip install ccxt

# Verify installation
echo "🔍 Verifying installations..."
python -c "import pandas; import numpy; import matplotlib; import requests; import dotenv; import pandas_ta; print('✅ All required packages are now installed!')"

echo ""
echo "🚀 Environment setup complete! You can now run the test script again:"
echo "python test_environment.py"
