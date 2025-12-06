"""
Test forecasting functionality after fixes
"""
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add dashboard modules to path
sys.path.append(str(Path(__file__).parent / 'visualization' / 'dashboard'))

import pandas as pd
import numpy as np
from ml_integration import MLModelIntegrator

def test_forecast():
    """Test that forecasting works correctly"""
    print("=" * 60)
    print("Testing Forecasting Model Fix")
    print("=" * 60)

    # Initialize integrator
    integrator = MLModelIntegrator()

    # Load forecasting model
    print("\n1. Loading forecasting model...")
    rf_loaded = integrator.load_forecasting_model()

    if not rf_loaded:
        print("   ✗ Failed to load forecasting model")
        return False

    print(f"   ✓ Model loaded successfully")
    print(f"   ✓ Model type: {type(integrator.rf_model)}")
    print(f"   ✓ Lookback days: {integrator.lookback_days}")
    print(f"   ✓ Forecast horizon: {integrator.forecast_horizon}")

    # Create sample data
    print("\n2. Creating sample data...")
    dates = pd.date_range(end='2024-01-31', periods=90, freq='D')
    counts = np.random.randint(100, 300, size=90)

    df = pd.DataFrame({
        'timestamp': dates,
        'count': counts
    })

    print(f"   ✓ Created {len(df)} days of sample data")

    # Test forecast generation
    print("\n3. Testing forecast generation...")
    try:
        forecast_df = integrator.generate_forecast(df, days_ahead=14)
        print(f"   ✓ Forecast generated successfully")
        print(f"   ✓ Forecast length: {len(forecast_df)} days")
        print(f"   ✓ Columns: {forecast_df.columns.tolist()}")
        print(f"\n   First 5 forecast values:")
        print(forecast_df.head().to_string(index=False))
        return True
    except Exception as e:
        print(f"   ✗ Forecast generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forecast()
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed! Forecasting is working correctly.")
    else:
        print("✗ Tests failed. Please check the errors above.")
    print("=" * 60)
    sys.exit(0 if success else 1)
