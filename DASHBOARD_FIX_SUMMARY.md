# Dashboard Fix Summary

## Issue
The dashboard was getting this error when trying to forecast:
```
RuntimeError: Failed to generate forecast: 'dict' object has no attribute 'predict'
```

## Root Cause
The forecasting model is stored as a **dictionary** with the following structure:
```python
{
    'model': RandomForestRegressor,  # The actual model
    'lookback_days': 30,
    'forecast_horizon': 7
}
```

The code was trying to use the entire dictionary as the model, instead of extracting the actual `RandomForestRegressor` from `model_data['model']`.

## Solution
Updated `ml_integration.py` to:

1. **Extract the actual model from the dictionary**:
   - Check if loaded data is a dictionary with a `'model'` key
   - Extract `model_data['model']` as the actual RandomForest model
   - Store `lookback_days` and `forecast_horizon` as instance variables

2. **Handle both formats**:
   - If the model is stored as a dictionary (current format), extract it
   - If the model is stored directly (fallback), use it as-is

3. **Use stored parameters**:
   - Updated `_predict_with_sequence_model()` to use `self.lookback_days` and `self.forecast_horizon` instead of hardcoded values

## Files Modified

### 1. visualization/dashboard/ml_integration.py

#### Added instance variables:
```python
def __init__(self):
    self.rf_model = None
    self.anomaly_model = None
    self.anomaly_detector = None
    self.outage_model = None
    # Forecasting model parameters
    self.lookback_days = 30
    self.forecast_horizon = 7
```

#### Updated load_forecasting_model():
```python
# Load the model (stored as a dict with 'model', 'lookback_days', 'forecast_horizon')
model_data = joblib.load(model_path)

# Extract the actual model from the dictionary if needed
if isinstance(model_data, dict) and 'model' in model_data:
    self.rf_model = model_data['model']
    self.lookback_days = model_data.get('lookback_days', 30)
    self.forecast_horizon = model_data.get('forecast_horizon', 7)
else:
    # Fallback for direct model format
    self.rf_model = model_data
    self.lookback_days = 30
    self.forecast_horizon = 7
```

#### Updated _predict_with_sequence_model():
```python
lookback_days = self.lookback_days
forecast_horizon = self.forecast_horizon
```

## Previous Fixes (From Earlier)

### Path Resolution
Also fixed model paths to use absolute paths from project root:
- Models are now loaded from `/ml_models` folder using `PROJECT_ROOT`
- Works correctly regardless of where the dashboard is run from

## Test Results

✓ All 3 models load successfully:
- RandomForest Forecaster (from dictionary format)
- Isolation Forest Anomaly Detector
- K-Means Outage Clustering

✓ Model prediction test:
- Model type: `sklearn.ensemble._forest.RandomForestRegressor`
- Can predict successfully
- Returns correct shape: `(1, 7)` for 7-day forecasts

## How to Run the Dashboard

```bash
cd "c:\Users\muqku\OneDrive\Desktop\DsdeProject\2110403-DSDE-M150-Lover"
streamlit run visualization/dashboard/main_dashboard.py
```

The dashboard should now work correctly without the "dict object has no attribute 'predict'" error!
