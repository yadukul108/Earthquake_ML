from pathlib import Path
import sys

import joblib
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

required = [
    ROOT / "app" / "app.py",
    ROOT / "app" / "utils.py",
    ROOT / "models" / "pga_mlp_development.pt",
    ROOT / "models" / "pga_scaler.pkl",
    ROOT / "models" / "pga_imputer.pkl",
    ROOT / "data" / "processed" / "test_data.csv",
    ROOT / "data" / "processed" / "final_test_predictions.csv",
]

print("Checking required project artifacts...\n")

failed = False

for path in required:
    ok = path.exists()
    print(("✓ " if ok else "✗ ") + str(path.relative_to(ROOT)))
    failed |= not ok

if failed:
    print("\nMissing required artifacts.")
    sys.exit(1)

print("\nLoading model artifacts...")

model_state = torch.load(
    ROOT / "models" / "pga_mlp_development.pt",
    map_location="cpu",
    weights_only=False,
)

scaler = joblib.load(ROOT / "models" / "pga_scaler.pkl")
imputer = joblib.load(ROOT / "models" / "pga_imputer.pkl")

assert isinstance(model_state, dict)
assert getattr(scaler, "n_features_in_", 13) == 13
assert getattr(imputer, "n_features_in_", 13) == 13

print("✓ PyTorch checkpoint")
print("✓ StandardScaler")
print("✓ SimpleImputer")

test = pd.read_csv(ROOT / "data" / "processed" / "test_data.csv")
pred = pd.read_csv(ROOT / "data" / "processed" / "final_test_predictions.csv")

assert len(test) == len(pred)

assert test["event_id"].reset_index(drop=True).equals(
    pred["event_id"].reset_index(drop=True)
)

print(f"✓ Test rows: {len(test):,}")
print(f"✓ Event IDs aligned: True")

print("\nProject validation PASSED.")
