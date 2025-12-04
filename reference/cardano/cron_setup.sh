#!/bin/bash

# Project Solaris - Cardano Trading System
# Cron Job Setup Script

echo "🕒 Setting up automatic data collection for Cardano Trading Bot"

# Get the full path to the project directory
PROJECT_DIR=$(pwd)
echo "📂 Project directory: $PROJECT_DIR"

# Validate the environment
if [ ! -f "$PROJECT_DIR/cardano_data.py" ]; then
    echo "❌ Error: cardano_data.py not found in $PROJECT_DIR"
    exit 1
fi

# Ensure data directory exists
mkdir -p "$PROJECT_DIR/data/cardano"
mkdir -p "$PROJECT_DIR/logs"

echo "📁 Data directories ready"

# Determine the Python executable path within the conda environment
CONDA_ENV="cardano"

# Determine the location of the conda executable
if command -v conda &> /dev/null; then
    CONDA_PATH=$(command -v conda)
    echo "✅ Found conda at: $CONDA_PATH"
else
    echo "❌ conda not found. Please make sure conda is installed and in your PATH."
    exit 1
fi

# Get conda base directory
CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
echo "📁 Conda base directory: $CONDA_BASE"

# Check if the environment exists
if conda env list | grep -q "$CONDA_ENV"; then
    echo "✅ Found conda environment: $CONDA_ENV"
else
    echo "❌ Conda environment '$CONDA_ENV' not found. Please create it first."
    exit 1
fi

# Get the Python path from the conda environment
PYTHON_PATH="$CONDA_BASE/envs/$CONDA_ENV/bin/python"
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Python executable not found at $PYTHON_PATH"
    # Try alternate location
    PYTHON_PATH="$CONDA_BASE/envs/$CONDA_ENV/python"
    if [ ! -f "$PYTHON_PATH" ]; then
        echo "❌ Python executable not found in conda environment. Please check your installation."
        exit 1
    fi
fi

echo "✅ Found Python executable: $PYTHON_PATH"

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
conda activate $CONDA_ENV
$PYTHON_PATH $PROJECT_DIR/cardano_data.py --weeks 1
EOL

# Make the script executable
chmod +x "$CRON_SCRIPT"
echo "✅ Created cron runner script: $CRON_SCRIPT"

# Create the cron job to run every day at 1:30 AM
CRON_JOB="30 1 * * * $CRON_SCRIPT >> $PROJECT_DIR/logs/cron.log 2>&1"

# Check if cron job already exists
EXISTING_CRON=$(crontab -l 2>/dev/null | grep -F "$CRON_SCRIPT")

if [ -z "$EXISTING_CRON" ]; then
    echo "Adding cron job to run daily at 1:30 AM..."
    
    # Save existing crontab
    crontab -l 2>/dev/null > crontab_temp
    
    # Add new job
    echo "# Project Solaris Cardano Data Fetcher - Daily update at 1:30 AM" >> crontab_temp
    echo "$CRON_JOB" >> crontab_temp
    
    # Install new crontab
    crontab crontab_temp
    rm crontab_temp
    
    echo "✅ Cron job installed successfully!"
else
    echo "✅ Cron job already exists"
fi

echo ""
echo "📋 To verify your cron job, run: crontab -l"
echo "🧪 To test the data fetcher manually, run:"
echo "bash $CRON_SCRIPT"
echo ""
echo "🚀 Setup complete! Your system will automatically update Cardano data daily at 1:30 AM."
