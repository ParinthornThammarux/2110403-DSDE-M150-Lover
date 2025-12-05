"""
Test script to verify dashboard structure without running Streamlit
"""

import sys
from pathlib import Path

print("=" * 60)
print("Dashboard Structure Test")
print("=" * 60)

# Check file existence
files_to_check = [
    'main_dashboard.py',
    'viz_modules.py',
    'ml_integration.py',
    'README.md',
    'requirements.txt'
]

print("\n[Files] Checking files:")
for file in files_to_check:
    exists = Path(file).exists()
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {file}")

# Check data file
print("\n[Data] Checking data files:")
data_files = [
    '../../clean_data.csv',
    '../../ml_models/forecasting/models/rf_forecaster.pkl',
    '../../ml_models/anomaly_detection/anomaly_if_model.pkl'
]

for file in data_files:
    exists = Path(file).exists()
    status = "[OK]" if exists else "[WARNING]"
    print(f"{status} {file}")

print("\n" + "=" * 60)
print("Structure check complete!")
print("=" * 60)

print("\n[INFO] To run the dashboard:")
print("   1. Install dependencies: pip install -r requirements.txt")
print("   2. Run: streamlit run main_dashboard.py")
print("\n   Or from root: streamlit run visualization/dashboard/main_dashboard.py")
