"""
Evaluation utilities for earthquake ground-motion models.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def regression_metrics(y_true, y_pred):
    """Calculate RMSE, MAE and R²."""

    return {
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),
        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),
        "R2": r2_score(
            y_true,
            y_pred
        )
    }
