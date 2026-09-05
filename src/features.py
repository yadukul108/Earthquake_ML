"""
Feature engineering utilities for earthquake ground-motion prediction.
"""

def get_final_features():
    return [
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
        "grid_lon"
    ]
