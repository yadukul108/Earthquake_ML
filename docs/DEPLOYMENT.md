# Deployment Guide

## Local

From the project root:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Confirm `app/app.py` is committed.
3. Confirm `models/` contains the three required artifacts.
4. Confirm required processed CSV files are committed or otherwise available.
5. Create a Streamlit Community Cloud app.
6. Select the GitHub repository.
7. Set the main file to:

```text
app/app.py
```

8. Deploy.

## Required model artifacts

```text
models/pga_mlp_development.pt
models/pga_scaler.pkl
models/pga_imputer.pkl
```

## Critical serialization detail

The scikit-learn artifacts were saved using joblib.

Load them using:

```python
import joblib

scaler = joblib.load("models/pga_scaler.pkl")
imputer = joblib.load("models/pga_imputer.pkl")
```

Do not replace this with `pickle.load()`.

The PyTorch checkpoint is loaded using:

```python
torch.load(
    "models/pga_mlp_development.pt",
    map_location="cpu",
    weights_only=False
)
```

## Deployment sanity test

Use:

```text
Magnitude             5.5
Depth                 20
Epicentral distance   100
Hypocentral distance  105
Vs30                  500
Latitude              20
Longitude             78
```

The current model should produce approximately:

```text
log-PGA ≈ -0.0918
PGA ≈ 0.8094
```

Small floating-point differences are acceptable.
