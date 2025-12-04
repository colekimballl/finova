#!/bin/bash

# Script to fix pandas-ta compatibility issue with numpy
# This fixes the "ImportError: cannot import name 'NaN' from 'numpy'" error

echo "🔧 Fixing pandas-ta compatibility issue..."

# Find the problematic file
SQUEEZE_PRO_FILE=$(find /usr/local/anaconda3/envs/cardano/lib/python3.9/site-packages/pandas_ta -name "squeeze_pro.py")

if [ -z "$SQUEEZE_PRO_FILE" ]; then
    echo "❌ Could not find the squeeze_pro.py file. Make sure pandas-ta is installed."
    exit 1
fi

echo "✅ Found file: $SQUEEZE_PRO_FILE"

# Create a backup of the original file
cp "$SQUEEZE_PRO_FILE" "${SQUEEZE_PRO_FILE}.bak"
echo "✅ Created backup at ${SQUEEZE_PRO_FILE}.bak"

# Replace the problematic import
sed -i.tmp 's/from numpy import NaN as npNaN/import numpy as np\nnpNaN = np.nan/' "$SQUEEZE_PRO_FILE"
echo "✅ Modified import statement in $SQUEEZE_PRO_FILE"

# Test if the fix worked
echo "🔍 Testing if the fix worked..."
python -c "import pandas_ta; print('✅ Success! pandas_ta can now be imported.')"

if [ $? -eq 0 ]; then
    echo "🚀 Fix successful! You can now use pandas-ta in your scripts."
else
    echo "❌ Fix unsuccessful. Restoring backup..."
    cp "${SQUEEZE_PRO_FILE}.bak" "$SQUEEZE_PRO_FILE"
    echo "⚠️ You may need to reinstall an older version of pandas-ta that's compatible with your numpy version."
    echo "   Try: pip uninstall pandas-ta && pip install pandas-ta==0.3.14b0"
fi
