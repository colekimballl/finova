#!/bin/bash
# cleanup.sh - Script to clean up redundant files
# This script will remove duplicate and obsolete setup files

# Define colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Files to be consolidated or removed
OLD_FILES=(
  "conda_setup.sh"
  "conda_setup_2.sh"
  "fix_pandas_ta.sh"
  "test_environment_simple.py"
  "coinbase_data.py"
  "coinbase_data1.py"
)

# Ensure we have the new consolidated files first
if [[ ! -f "setup.sh" || ! -f "test_environment.py" ]]; then
  echo -e "${RED}Error: New consolidated files (setup.sh or test_environment.py) not found.${NC}"
  echo -e "${YELLOW}Please create these files first before cleaning up.${NC}"
  exit 1
fi

# Create backup directory
BACKUP_DIR="junk/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo -e "${BLUE}Created backup directory: $BACKUP_DIR${NC}"

# Move old files to backup
for file in "${OLD_FILES[@]}"; do
  if [[ -f "$file" ]]; then
    echo -e "${YELLOW}Moving $file to backup...${NC}"
    mv "$file" "$BACKUP_DIR/"
    echo -e "${GREEN}✓ $file moved to backup${NC}"
  else
    echo -e "${BLUE}File $file not found, skipping...${NC}"
  fi
done

# Check if there's a conflict with cardano_data.py vs. coinbase_data.py
if [[ -f "cardano_data.py" ]]; then
  echo -e "${GREEN}Found cardano_data.py - this will be the main data fetcher${NC}"
else
  echo -e "${YELLOW}cardano_data.py not found. Restoring from backup...${NC}"
  if [[ -f "$BACKUP_DIR/coinbase_data.py" ]]; then
    cp "$BACKUP_DIR/coinbase_data.py" "./cardano_data.py"
    echo -e "${GREEN}Restored coinbase_data.py as cardano_data.py${NC}"
  else
    echo -e "${RED}No data fetcher file found in backup. Please check manually.${NC}"
  fi
fi

# Make the new scripts executable
chmod +x setup.sh
if [[ -f "run_data_update.sh" ]]; then
  chmod +x run_data_update.sh
fi

echo -e "${GREEN}Clean up complete!${NC}"
echo -e "${BLUE}Old files have been backed up to: $BACKUP_DIR${NC}"
echo -e "${BLUE}Use the following files going forward:${NC}"
echo -e "  - setup.sh           (for environment and cron setup)"
echo -e "  - test_environment.py (for checking your environment)"
echo -e "  - cardano_data.py    (for data collection)"
echo -e "  - .env               (for configuration)"
