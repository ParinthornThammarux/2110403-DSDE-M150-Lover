"""
Test script to verify ML models can be loaded from ml_models folder
"""
import sys
from pathlib import Path
import joblib

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Get project root directory
PROJECT_ROOT = Path(__file__).parent

def test_model_loading():
    """Test that all models load correctly from ml_models folder"""
    print("=" * 60)
    print("Testing ML Model Loading from /ml_models folder")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")

    # Test forecasting model
    print("\n1. Testing Forecasting Model...")
    forecasting_path = PROJECT_ROOT / 'ml_models' / 'forecasting' / 'models' / 'rf_forecaster.pkl'
    print(f"   Path: {forecasting_path}")
    rf_loaded = False
    if forecasting_path.exists():
        try:
            rf_model = joblib.load(str(forecasting_path))
            rf_loaded = True
            print("   ✓ RandomForest Forecaster loaded successfully")
            print(f"   ✓ Model type: {type(rf_model)}")
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
    else:
        print("   ✗ Model file not found")

    # Test anomaly detection model
    print("\n2. Testing Anomaly Detection Model...")
    anomaly_path = PROJECT_ROOT / 'ml_models' / 'anomaly_detection' / 'anomaly_if_model.pkl'
    print(f"   Path: {anomaly_path}")
    anomaly_loaded = False
    if anomaly_path.exists():
        try:
            anomaly_model = joblib.load(str(anomaly_path))
            anomaly_loaded = True
            print("   ✓ Isolation Forest Anomaly Detector loaded successfully")
            print(f"   ✓ Model type: {type(anomaly_model)}")
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
    else:
        print("   ✗ Model file not found")

    # Test outage clustering model
    print("\n3. Testing Outage Clustering Model...")
    outage_path = PROJECT_ROOT / 'ml_models' / 'outage_model' / 'models' / 'outage_kmeans_model.pkl'
    print(f"   Path: {outage_path}")
    outage_loaded = False
    if outage_path.exists():
        try:
            outage_model = joblib.load(str(outage_path))
            outage_loaded = True
            print("   ✓ K-Means Outage Clustering loaded successfully")
            print(f"   ✓ Model type: {type(outage_model)}")
        except Exception as e:
            print(f"   ✗ Failed to load: {e}")
    else:
        print("   ✗ Model file not found")

    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    total_models = 3
    loaded_models = sum([rf_loaded, anomaly_loaded, outage_loaded])
    print(f"  Models loaded: {loaded_models}/{total_models}")

    if loaded_models == total_models:
        print("  ✓ All models loaded successfully!")
        print("  ✓ Dashboard should work correctly")
    else:
        print(f"  ⚠ {total_models - loaded_models} model(s) failed to load")
        print("  ⚠ Some dashboard features may not work")

    print("=" * 60)

    return loaded_models == total_models

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
