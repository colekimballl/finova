#!/bin/bash
# Project Solaris AI - Cardano Trading System
# Consolidated Setup Script
# This script handles all setup tasks:
# 1. Conda environment setup
# 2. Package installation
# 3. Fixing pandas-ta compatibility issues
# 4. Setting up the cron job

# Define colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Display header
echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}  Project Solaris AI - Cardano Trading System  ${NC}"
echo -e "${BLUE}==============================================${NC}"

# Get the full path to the project directory
PROJECT_DIR=$(pwd)
print_info "Project directory: $PROJECT_DIR"

# Define environment name
ENV_NAME="cardano"

# -------------- PART 1: CONDA ENVIRONMENT SETUP --------------

print_info "STEP 1: Setting up conda environment"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    print_warning "Environment '$ENV_NAME' already exists. Do you want to remove it and create a fresh one? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        conda env remove -n "$ENV_NAME"
        print_info "Creating new conda environment: $ENV_NAME"
        conda create -n "$ENV_NAME" python=3.9 -y
    else
        print_warning "Using existing environment."
    fi
else
    print_info "Creating new conda environment: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.9 -y
fi

# Get conda base directory
CONDA_BASE=$(conda info --base)

# -------------- PART 2: PACKAGE INSTALLATION --------------

print_info "STEP 2: Installing required packages"

# Create a temporary script to install packages
TMP_INSTALL_SCRIPT="$PROJECT_DIR/tmp_install.sh"
cat > "$TMP_INSTALL_SCRIPT" << EOL
#!/bin/bash
# Activate environment
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install required packages with conda
conda install -y pandas numpy matplotlib requests
conda install -y -c conda-forge ccxt

# Install additional packages with pip
pip install python-dotenv pandas-ta
EOL

# Make the script executable
chmod +x "$TMP_INSTALL_SCRIPT"

# Run the installation script
print_info "Installing packages... this may take a few minutes"
"$TMP_INSTALL_SCRIPT"

# Remove temporary script
rm "$TMP_INSTALL_SCRIPT"

# -------------- PART 3: FIX PANDAS-TA COMPATIBILITY --------------

print_info "STEP 3: Fixing pandas-ta compatibility issues"

# Create a temporary script to fix pandas-ta
TMP_FIX_SCRIPT="$PROJECT_DIR/tmp_fix_pandas_ta.sh"
cat > "$TMP_FIX_SCRIPT" << EOL
#!/bin/bash
# Activate environment
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Find the problematic file
SQUEEZE_PRO_FILE=\$(find "$CONDA_BASE/envs/$ENV_NAME/lib/python3.9/site-packages/pandas_ta" -name "squeeze_pro.py" 2>/dev/null)

if [ -z "\$SQUEEZE_PRO_FILE" ]; then
    echo "Could not find the squeeze_pro.py file. Make sure pandas-ta is installed."
    exit 1
fi

echo "Found file: \$SQUEEZE_PRO_FILE"

# Create a backup of the original file
cp "\$SQUEEZE_PRO_FILE" "\${SQUEEZE_PRO_FILE}.bak"
echo "Created backup at \${SQUEEZE_PRO_FILE}.bak"

# Replace the problematic import
sed -i.tmp 's/from numpy import NaN as npNaN/import numpy as np\\nnpNaN = np.nan/' "\$SQUEEZE_PRO_FILE"
echo "Modified import statement in \$SQUEEZE_PRO_FILE"

# Test if the fix worked
python -c "import pandas_ta; print('Success! pandas_ta can now be imported.')"
EOL

# Make the script executable
chmod +x "$TMP_FIX_SCRIPT"

# Run the fix script
"$TMP_FIX_SCRIPT"

# Remove temporary script
rm "$TMP_FIX_SCRIPT"

# -------------- PART 4: DIRECTORY SETUP --------------

print_info "STEP 4: Setting up project directories"

# Create required directories
mkdir -p "$PROJECT_DIR/data/cardano"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/strategies"
mkdir -p "$PROJECT_DIR/backtest"

print_success "Directories created"

# -------------- PART 5: CRON JOB SETUP --------------

print_info "STEP 5: Setting up automated data collection"

# Determine the Python executable path within the conda environment
PYTHON_PATH="$CONDA_BASE/envs/$ENV_NAME/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    # Try alternate location
    PYTHON_PATH="$CONDA_BASE/envs/$ENV_NAME/python"
    if [ ! -f "$PYTHON_PATH" ]; then
        print_error "Python executable not found in conda environment. Skipping cron setup."
        PYTHON_PATH=""
    fi
fi

if [ -n "$PYTHON_PATH" ]; then
    print_success "Found Python executable: $PYTHON_PATH"
    
    # Create a shell script that will be run by cron
    CRON_SCRIPT="$PROJECT_DIR/run_data_update.sh"
    cat > "$CRON_SCRIPT" << EOL
#!/bin/bash
# Script to run Cardano data update from cron
# Created: $(date)

# Change to project directory
cd $PROJECT_DIR

# Activate conda environment and run data update
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $ENV_NAME
$PYTHON_PATH $PROJECT_DIR/cardano_data.py --weeks 1
EOL

    # Make the script executable
    chmod +x "$CRON_SCRIPT"
    print_success "Created cron runner script: $CRON_SCRIPT"
    
    # Create the cron job to run every day at 1:30 AM
    CRON_JOB="30 1 * * * $CRON_SCRIPT >> $PROJECT_DIR/logs/cron.log 2>&1"
    
    # Check if cron job already exists
    EXISTING_CRON=$(crontab -l 2>/dev/null | grep -F "$CRON_SCRIPT")
    
    if [ -z "$EXISTING_CRON" ]; then
        print_info "Would you like to install the cron job to run daily at 1:30 AM? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            # Save existing crontab
            crontab -l 2>/dev/null > crontab_temp
            
            # Add new job
            echo "# Project Solaris Cardano Data Fetcher - Daily update at 1:30 AM" >> crontab_temp
            echo "$CRON_JOB" >> crontab_temp
            
            # Install new crontab
            crontab crontab_temp
            rm crontab_temp
            
            print_success "Cron job installed successfully!"
        else
            print_warning "Cron job installation skipped. You can run it manually with: bash $CRON_SCRIPT"
        fi
    else
        print_success "Cron job already exists"
    fi
fi

# -------------- PART 6: FINAL STEPS --------------

print_info "STEP 6: Creating .env template if it doesn't exist"

# Create .env template if it doesn't exist
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cat > "$PROJECT_DIR/.env" << EOL
# Coinbase API credentials
COINBASE_API_KEY="organizations/{org_id}/apiKeys/{key_id}"
COINBASE_API_SECRET="your-secret-key-here"

# Data Collection Settings
HISTORICAL_WEEKS=52  # How many weeks of historical data to fetch initially
DATA_DIRECTORY="data/cardano"

# Trading Parameters
DEFAULT_TIMEFRAME="1h"  # Default timeframe for trading strategy
POSITION_SIZE=10  # Size in USD for each trade
PROFIT_TARGET=0.05  # 5% profit target
STOP_LOSS=0.03  # 3% stop loss
MAX_RISK_PER_TRADE=5  # Maximum percentage of account to risk per trade
EOL
    print_success "Created .env template file. Please edit it with your actual API credentials."
else
    print_success ".env file already exists"
fi

# -------------- FINAL SUMMARY --------------
echo ""
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}     Project Solaris AI Setup Complete!        ${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo -e "${BLUE}To activate the conda environment:${NC}"
echo -e "conda activate $ENV_NAME"
echo ""
echo -e "${BLUE}To test your data connection:${NC}"
echo -e "python $PROJECT_DIR/test_environment.py"
echo ""
echo -e "${BLUE}To manually update data:${NC}"
echo -e "bash $CRON_SCRIPT"
echo ""
echo -e "${BLUE}To check your cron job:${NC}"
echo -e "crontab -l"
echo ""
echo -e "${GREEN}🚀 Project Solaris AI is ready for liftoff!${NC}"
