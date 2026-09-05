from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import streamlit as st


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"


# =========================================================
# FINAL FEATURE ORDER
# =========================================================

FEATURE_ORDER = [
    "magnitude",
    "event_depth",
    "epicentral_distance_km",
    "hypocentral_distance_km",
    "log_distance",
    "magnitude_squared",
    "magnitude_distance_interaction",
    "depth_distance_interaction",
    "Vs30_clean",
    "log_Vs30",
    "Vs30_available",
    "grid_lat",
    "grid_lon",
]


# =========================================================
# FINAL MLP ARCHITECTURE
#
# 13 → 64 → 32 → 16 → 1
#
# This architecture matches:
#
# network.0.weight       (64, 13)
# network.2             BatchNorm1d(64)
# network.4.weight       (32, 64)
# network.6             BatchNorm1d(32)
# network.8.weight       (16, 32)
# network.10.weight      (1, 16)
# =========================================================

class PGA_MLP(nn.Module):

    def __init__(self, input_dim=13):

        super().__init__()

        self.network = nn.Sequential(

            # 13 → 64
            nn.Linear(input_dim, 64),
            nn.ReLU(),

            # BatchNorm(64)
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # 64 → 32
            nn.Linear(64, 32),
            nn.ReLU(),

            # BatchNorm(32)
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # 32 → 16
            nn.Linear(32, 16),
            nn.ReLU(),

            # 16 → 1
            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# GENERIC CSV LOADER
# =========================================================

@st.cache_data
def load_csv(filename):

    path = DATA_DIR / filename

    if not path.exists():

        return None

    return pd.read_csv(path)


# =========================================================
# DATA LOADERS
# =========================================================

def load_test_predictions():

    return load_csv(
        "final_test_predictions.csv"
    )


def load_test_data():

    return load_csv(
        "test_data.csv"
    )


def load_test_metrics():

    return load_csv(
        "final_test_metrics.csv"
    )


def load_event_results():

    return load_csv(
        "final_test_per_event_results.csv"
    )


def load_residual_summary():

    return load_csv(
        "final_test_event_residual_summary.csv"
    )


def load_feature_importance():

    return load_csv(
        "mlp_permutation_feature_importance.csv"
    )


def load_response_summary():

    return load_csv(
        "mlp_response_summary.csv"
    )


def load_response_depth():

    return load_csv(
        "mlp_response_depth.csv"
    )


def load_response_distance():

    return load_csv(
        "mlp_response_hypocentral_distance.csv"
    )


def load_response_magnitude():

    return load_csv(
        "mlp_response_magnitude.csv"
    )


def load_response_vs30():

    return load_csv(
        "mlp_response_Vs30.csv"
    )


def load_model_comparison():

    return load_csv(
        "final_model_comparison.csv"
    )


def load_model_results():

    return load_csv(
        "final_model_results_table.csv"
    )


# =========================================================
# LOAD SCALER
#
# IMPORTANT:
# These .pkl files were created using joblib.
# Therefore use joblib.load(), NOT pickle.load().
# =========================================================

@st.cache_resource
def load_scaler():

    path = MODEL_DIR / "pga_scaler.pkl"

    if not path.exists():

        raise FileNotFoundError(
            f"Scaler not found:\n{path}"
        )

    scaler = joblib.load(path)

    return scaler


# =========================================================
# LOAD IMPUTER
# =========================================================

@st.cache_resource
def load_imputer():

    path = MODEL_DIR / "pga_imputer.pkl"

    if not path.exists():

        raise FileNotFoundError(
            f"Imputer not found:\n{path}"
        )

    imputer = joblib.load(path)

    return imputer


# =========================================================
# LOAD FINAL PYTORCH MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_path = (
        MODEL_DIR /
        "pga_mlp_development.pt"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{model_path}"
        )

    # Explicitly use weights_only=False because
    # the saved file is a state_dict checkpoint.
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    model = PGA_MLP(
        input_dim=len(FEATURE_ORDER)
    )

    # The current checkpoint is an OrderedDict
    # containing the state_dict directly.
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# =========================================================
# VALIDATE MODEL ARTIFACTS
# =========================================================

@st.cache_resource
def validate_model_artifacts():

    scaler = load_scaler()

    imputer = load_imputer()

    model = load_model()

    # -----------------------------------------------------
    # Check scaler
    # -----------------------------------------------------

    if hasattr(
        scaler,
        "n_features_in_"
    ):

        if scaler.n_features_in_ != 13:

            raise ValueError(
                "Scaler expects "
                f"{scaler.n_features_in_} features, "
                "but final MLP requires 13."
            )

    # -----------------------------------------------------
    # Check imputer
    # -----------------------------------------------------

    if hasattr(
        imputer,
        "n_features_in_"
    ):

        if imputer.n_features_in_ != 13:

            raise ValueError(
                "Imputer expects "
                f"{imputer.n_features_in_} features, "
                "but final MLP requires 13."
            )

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    first_layer = (
        model.network[0]
    )

    if first_layer.in_features != 13:

        raise ValueError(
            "MLP expects "
            f"{first_layer.in_features} features, "
            "but FEATURE_ORDER contains 13."
        )

    return True


# =========================================================
# LIVE FEATURE ENGINEERING
# =========================================================

def build_prediction_row(
    magnitude,
    event_depth,
    epicentral_distance_km,
    hypocentral_distance_km,
    vs30,
    grid_lat,
    grid_lon
):

    magnitude = float(magnitude)

    event_depth = float(
        event_depth
    )

    epicentral_distance_km = float(
        epicentral_distance_km
    )

    hypocentral_distance_km = float(
        hypocentral_distance_km
    )

    vs30 = float(vs30)

    grid_lat = float(
        grid_lat
    )

    grid_lon = float(
        grid_lon
    )

    # -----------------------------------------------------
    # log distance
    # -----------------------------------------------------

    log_distance = np.log10(
        max(
            hypocentral_distance_km,
            1e-6
        )
    )

    # -----------------------------------------------------
    # magnitude squared
    # -----------------------------------------------------

    magnitude_squared = (
        magnitude ** 2
    )

    # -----------------------------------------------------
    # magnitude × distance
    # -----------------------------------------------------

    magnitude_distance_interaction = (
        magnitude *
        log_distance
    )

    # -----------------------------------------------------
    # depth × distance
    # -----------------------------------------------------

    depth_distance_interaction = (
        event_depth *
        log_distance
    )

    # -----------------------------------------------------
    # Vs30
    # -----------------------------------------------------

    if (
        np.isfinite(vs30)
        and vs30 > 0
    ):

        vs30_clean = vs30

        log_vs30 = np.log10(
            max(
                vs30,
                1e-6
            )
        )

        vs30_available = 1.0

    else:

        vs30_clean = np.nan

        log_vs30 = np.nan

        vs30_available = 0.0

    # -----------------------------------------------------
    # Build row
    # -----------------------------------------------------

    row = pd.DataFrame(
        [[
            magnitude,
            event_depth,
            epicentral_distance_km,
            hypocentral_distance_km,
            log_distance,
            magnitude_squared,
            magnitude_distance_interaction,
            depth_distance_interaction,
            vs30_clean,
            log_vs30,
            vs30_available,
            grid_lat,
            grid_lon
        ]],
        columns=FEATURE_ORDER
    )

    return row


# =========================================================
# LIVE PGA PREDICTION
# =========================================================

def predict_log_pga(
    magnitude,
    event_depth,
    epicentral_distance_km,
    hypocentral_distance_km,
    vs30,
    grid_lat,
    grid_lon
):

    # -----------------------------------------------------
    # Validate all model artifacts
    # -----------------------------------------------------

    validate_model_artifacts()

    model = load_model()

    scaler = load_scaler()

    imputer = load_imputer()

    # -----------------------------------------------------
    # Feature engineering
    # -----------------------------------------------------

    row = build_prediction_row(
        magnitude=magnitude,
        event_depth=event_depth,
        epicentral_distance_km=
            epicentral_distance_km,
        hypocentral_distance_km=
            hypocentral_distance_km,
        vs30=vs30,
        grid_lat=grid_lat,
        grid_lon=grid_lon
    )

    # -----------------------------------------------------
    # Ensure correct feature order
    # -----------------------------------------------------

    row = row[
        FEATURE_ORDER
    ]

    # -----------------------------------------------------
    # Imputation
    # -----------------------------------------------------

    X_imputed = imputer.transform(
        row
    )

    # -----------------------------------------------------
    # Standardization
    # -----------------------------------------------------

    X_scaled = scaler.transform(
        X_imputed
    )

    # -----------------------------------------------------
    # Convert to PyTorch tensor
    # -----------------------------------------------------

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    model.eval()

    with torch.no_grad():

        prediction = (
            model(
                X_tensor
            )
            .cpu()
            .numpy()
            .ravel()[0]
        )

    predicted_log_pga = float(
        prediction
    )

    # Target is log10(PGA)
    predicted_pga = float(
        10 ** predicted_log_pga
    )

    return (
        predicted_log_pga,
        predicted_pga
    )


# =========================================================
# BUILD EARTHQUAKE EXPLORER DATA
# =========================================================

@st.cache_data
def build_explorer_data():

    predictions = (
        load_test_predictions()
    )

    test_data = (
        load_test_data()
    )

    # -----------------------------------------------------
    # File existence
    # -----------------------------------------------------

    if predictions is None:

        raise FileNotFoundError(
            "final_test_predictions.csv "
            "was not found."
        )

    if test_data is None:

        raise FileNotFoundError(
            "test_data.csv "
            "was not found."
        )

    # -----------------------------------------------------
    # Row count check
    # -----------------------------------------------------

    if len(predictions) != len(test_data):

        raise ValueError(
            "Prediction/test-data row mismatch: "
            f"{len(predictions)} vs "
            f"{len(test_data)}"
        )

    # -----------------------------------------------------
    # Required prediction columns
    # -----------------------------------------------------

    required_prediction_columns = [
        "event_id",
        "observed_log_PGA",
        "predicted_log_PGA",
    ]

    missing_prediction_columns = [
        col
        for col in required_prediction_columns
        if col not in predictions.columns
    ]

    if missing_prediction_columns:

        raise ValueError(
            "Missing prediction columns: "
            f"{missing_prediction_columns}"
        )

    # -----------------------------------------------------
    # Required test-data columns
    # -----------------------------------------------------

    required_test_columns = [
        "grid_lon",
        "grid_lat",
        "event_id",
        "magnitude",
        "event_lat",
        "event_lon",
        "epicentral_distance_km",
        "hypocentral_distance_km",
        "event_depth",
        "Vs30",
        "Vs30_clean",
        "site_class",
        "log_PGA",
    ]

    missing_test_columns = [
        col
        for col in required_test_columns
        if col not in test_data.columns
    ]

    if missing_test_columns:

        raise ValueError(
            "Missing test-data columns: "
            f"{missing_test_columns}"
        )

    # -----------------------------------------------------
    # Event alignment
    # -----------------------------------------------------

    event_aligned = (
        predictions["event_id"]
        .reset_index(drop=True)
        .equals(
            test_data["event_id"]
            .reset_index(drop=True)
        )
    )

    if not event_aligned:

        raise ValueError(
            "event_id ordering is not aligned "
            "between test_data and predictions."
        )

    # -----------------------------------------------------
    # Observed target alignment
    # -----------------------------------------------------

    observed_aligned = np.allclose(
        test_data["log_PGA"].to_numpy(),
        predictions[
            "observed_log_PGA"
        ].to_numpy(),
        equal_nan=True
    )

    if not observed_aligned:

        raise ValueError(
            "Observed log-PGA values are not aligned "
            "between test_data and predictions."
        )

    # -----------------------------------------------------
    # Build explorer dataframe
    # -----------------------------------------------------

    explorer = test_data[
        required_test_columns
    ].copy()

    # -----------------------------------------------------
    # Attach predictions
    # -----------------------------------------------------

    explorer[
        "observed_log_PGA"
    ] = predictions[
        "observed_log_PGA"
    ].to_numpy()

    explorer[
        "predicted_log_PGA"
    ] = predictions[
        "predicted_log_PGA"
    ].to_numpy()

    # -----------------------------------------------------
    # IMPORTANT
    #
    # final_test_predictions.csv contains:
    #
    # event_id
    # observed_log_PGA
    # predicted_log_PGA
    #
    # It does NOT contain residual.
    #
    # Therefore calculate:
    #
    # residual = observed - predicted
    # -----------------------------------------------------

    explorer[
        "residual"
    ] = (
        explorer[
            "observed_log_PGA"
        ]
        -
        explorer[
            "predicted_log_PGA"
        ]
    )

    return explorer