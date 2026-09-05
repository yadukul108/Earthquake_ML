import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# =========================================================
# IMPORT UTILS
# =========================================================

APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import (
    load_test_predictions,
    load_test_data,
    load_test_metrics,
    load_event_results,
    load_residual_summary,
    load_feature_importance,
    load_response_summary,
    load_response_depth,
    load_response_distance,
    load_response_magnitude,
    load_response_vs30,
    load_model_comparison,
    load_model_results,
    predict_log_pga,
    build_explorer_data,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Earthquake AI",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🌎 Earthquake AI")

    st.markdown(
        "**Ground-Motion Prediction**"
    )

    st.markdown(
        "Machine Learning + Deep Learning"
    )

    st.divider()

    st.markdown("**Navigation**")

    page = st.radio(
        "",
        [
            "🏠 Overview",
            "🌎 Earthquake Explorer",
            "🤖 Model Comparison",
            "🔍 Interpretability",
            "🎯 PGA Prediction",
        ],
    )

    st.divider()

    st.caption(
        "Earthquake Ground-Motion Prediction"
    )

    st.caption(
        "Research & portfolio application"
    )


# =========================================================
# LOAD ALL DATA
# =========================================================

@st.cache_data
def load_all_data():

    return {
        "predictions": load_test_predictions(),
        "test_data": load_test_data(),
        "metrics": load_test_metrics(),
        "events": load_event_results(),
        "residuals": load_residual_summary(),
        "importance": load_feature_importance(),
        "responses": load_response_summary(),
        "response_depth": load_response_depth(),
        "response_distance": load_response_distance(),
        "response_magnitude": load_response_magnitude(),
        "response_vs30": load_response_vs30(),
        "comparison": load_model_comparison(),
        "model_results": load_model_results(),
    }


data = load_all_data()


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.markdown(
        '<div class="main-title">'
        'Earthquake Ground-Motion Prediction'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Large-scale earthquake ground-motion prediction '
        'using machine learning and deep learning'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Observations", "2.87M")
    c2.metric("Earthquakes", "40")
    c3.metric("Grouped CV R²", "0.802")
    c4.metric("Unseen-Test R²", "0.610")

    st.divider()

    st.subheader("Project Objective")

    st.write(
        """
        This project predicts logarithmic Peak Ground
        Acceleration (log-PGA) using earthquake source
        parameters, propagation geometry, geographic
        information, and Vs30 site conditions.
        """
    )

    st.subheader(
        "Why earthquake-level validation?"
    )

    st.write(
        """
        Instead of randomly splitting millions of
        observations, entire earthquakes are kept together
        during validation. This provides a much stronger
        test of whether the model can generalize to
        earthquakes it has never seen before.
        """
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Training", "28 earthquakes")
    c2.metric("Validation", "6 earthquakes")
    c3.metric("Final Test", "6 earthquakes")

    st.divider()

    st.subheader("Final Model")

    st.success(
        """
        **PyTorch MLP**

        Grouped 5-fold CV:
        RMSE = 0.2128 | MAE = 0.1777 | R² = 0.8015

        Completely unseen test earthquakes:
        RMSE = 0.3113 | MAE = 0.2800 | R² = 0.6099
        """
    )

    st.subheader("Project Pipeline")

    st.write(
        """
        Earthquake data → Quality control →
        Feature engineering → Vs30 integration →
        Classical ML → Deep learning →
        Earthquake-grouped CV →
        Unseen-earthquake evaluation →
        Residual analysis → Interpretability
        """
    )

    st.divider()

    st.subheader("Key Result")

    st.info(
        """
        The PyTorch MLP achieved the strongest grouped
        cross-validation performance among the evaluated
        models, while retaining positive generalization
        performance on completely unseen earthquakes.
        """
    )


# =========================================================
# PAGE 2 — EARTHQUAKE EXPLORER
# =========================================================

elif page == "🌎 Earthquake Explorer":

    st.title("🌎 Earthquake Explorer")

    st.write(
        "Explore the model's performance on the six "
        "completely unseen earthquake events used for "
        "final testing."
    )

    try:
        explorer_data = build_explorer_data()

    except Exception as e:

        st.error(
            f"Could not build explorer data: {e}"
        )

        st.stop()

    if explorer_data is None:

        st.error(
            "Required test files were not found."
        )

        st.stop()

    events = sorted(
        explorer_data["event_id"]
        .dropna()
        .unique()
    )

    selected_event = st.selectbox(
        "Select an unseen earthquake",
        events,
    )

    event_data = explorer_data[
        explorer_data["event_id"] == selected_event
    ].copy()

    # -----------------------------------------------------
    # EVENT PARAMETERS
    # -----------------------------------------------------

    magnitude = float(
        event_data["magnitude"].iloc[0]
    )

    depth = float(
        event_data["event_depth"].iloc[0]
    )

    event_lat = float(
        event_data["event_lat"].iloc[0]
    )

    event_lon = float(
        event_data["event_lon"].iloc[0]
    )

    observations = len(event_data)

    # -----------------------------------------------------
    # PERFORMANCE
    # -----------------------------------------------------

    y_true = event_data[
        "observed_log_PGA"
    ].to_numpy()

    y_pred = event_data[
        "predicted_log_PGA"
    ].to_numpy()

    residual = y_true - y_pred

    rmse = float(
        np.sqrt(np.mean(residual ** 2))
    )

    mae = float(
        np.mean(np.abs(residual))
    )

    ss_res = np.sum(residual ** 2)

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = (
        float(1 - ss_res / ss_tot)
        if ss_tot != 0
        else np.nan
    )

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        f"""
        **Earthquake:** `{selected_event}`

        **Magnitude:** {magnitude:.1f}

        **Depth:** {depth:.2f} km

        **Epicenter:** ({event_lat:.3f}, {event_lon:.3f})
        """
    )

    # -----------------------------------------------------
    # PERFORMANCE KPIs
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Observations",
        f"{observations:,}"
    )

    c2.metric(
        "RMSE",
        f"{rmse:.4f}"
    )

    c3.metric(
        "MAE",
        f"{mae:.4f}"
    )

    c4.metric(
        "R²",
        f"{r2:.4f}"
    )

    # -----------------------------------------------------
    # PHYSICAL SUMMARY
    # -----------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    mean_distance = event_data[
        "hypocentral_distance_km"
    ].mean()

    mean_vs30 = event_data[
        "Vs30_clean"
    ].mean()

    vs30_availability = (
        event_data["Vs30_clean"]
        .notna()
        .mean()
    )

    mean_residual = event_data[
        "residual"
    ].mean()

    c1.metric(
        "Mean Distance",
        f"{mean_distance:.1f} km"
    )

    c2.metric(
        "Mean Vs30",
        f"{mean_vs30:.1f} m/s"
        if np.isfinite(mean_vs30)
        else "N/A"
    )

    c3.metric(
        "Vs30 Availability",
        f"{100 * vs30_availability:.1f}%"
    )

    c4.metric(
        "Mean Residual",
        f"{mean_residual:.4f}"
    )

    st.divider()

    # -----------------------------------------------------
    # MAP SAMPLE
    # -----------------------------------------------------

    map_data = event_data.copy()

    if len(map_data) > 12000:

        map_data = map_data.sample(
            n=12000,
            random_state=42,
        )

    # -----------------------------------------------------
    # MAP HELPER
    # -----------------------------------------------------

    def make_map(
        df,
        color_column,
        title,
    ):

        hover_columns = [
            "magnitude",
            "event_depth",
            "Vs30_clean",
            "hypocentral_distance_km",
        ]

        if hasattr(px, "scatter_map"):

            fig = px.scatter_map(
                df,
                lat="grid_lat",
                lon="grid_lon",
                color=color_column,
                hover_data=hover_columns,
                zoom=3,
                height=550,
                title=title,
            )

        else:

            fig = px.scatter_mapbox(
                df,
                lat="grid_lat",
                lon="grid_lon",
                color=color_column,
                hover_data=hover_columns,
                zoom=3,
                height=550,
                title=title,
            )

            fig.update_layout(
                mapbox_style="open-street-map"
            )

        fig.update_layout(
            margin=dict(
                l=0,
                r=0,
                t=50,
                b=0,
            )
        )

        return fig

    # -----------------------------------------------------
    # OBSERVED MAP
    # -----------------------------------------------------

    st.subheader(
        "Observed Ground Motion"
    )

    st.plotly_chart(
        make_map(
            map_data,
            "observed_log_PGA",
            f"Observed log-PGA — {selected_event}",
        ),
        use_container_width=True,
    )

    # -----------------------------------------------------
    # PREDICTED MAP
    # -----------------------------------------------------

    st.subheader(
        "Predicted Ground Motion"
    )

    st.plotly_chart(
        make_map(
            map_data,
            "predicted_log_PGA",
            f"Predicted log-PGA — {selected_event}",
        ),
        use_container_width=True,
    )

    # -----------------------------------------------------
    # RESIDUAL MAP
    # -----------------------------------------------------

    st.subheader(
        "Residual Ground Motion"
    )

    st.caption(
        "Residual = Observed log-PGA − Predicted log-PGA"
    )

    st.plotly_chart(
        make_map(
            map_data,
            "residual",
            f"Residual map — {selected_event}",
        ),
        use_container_width=True,
    )

    st.divider()

    # -----------------------------------------------------
    # OBSERVED VS PREDICTED
    # -----------------------------------------------------

    st.subheader(
        "Observed vs Predicted log-PGA"
    )

    scatter_data = event_data[
        [
            "observed_log_PGA",
            "predicted_log_PGA",
        ]
    ].dropna()

    if len(scatter_data) > 15000:

        scatter_data = scatter_data.sample(
            n=15000,
            random_state=42,
        )

    fig = px.scatter(
        scatter_data,
        x="observed_log_PGA",
        y="predicted_log_PGA",
        opacity=0.35,
        labels={
            "observed_log_PGA":
                "Observed log-PGA",
            "predicted_log_PGA":
                "Predicted log-PGA",
        },
        title=(
            f"Observed vs Predicted — {selected_event}"
        ),
    )

    minimum = min(
        scatter_data["observed_log_PGA"].min(),
        scatter_data["predicted_log_PGA"].min(),
    )

    maximum = max(
        scatter_data["observed_log_PGA"].max(),
        scatter_data["predicted_log_PGA"].max(),
    )

    fig.add_shape(
        type="line",
        x0=minimum,
        y0=minimum,
        x1=maximum,
        y1=maximum,
        line=dict(dash="dash"),
    )

    fig.update_layout(height=550)

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    # -----------------------------------------------------
    # EVENT DATA
    # -----------------------------------------------------

    with st.expander(
        "View event prediction data"
    ):

        st.dataframe(
            event_data[
                [
                    "event_id",
                    "grid_lat",
                    "grid_lon",
                    "magnitude",
                    "event_depth",
                    "hypocentral_distance_km",
                    "Vs30_clean",
                    "observed_log_PGA",
                    "predicted_log_PGA",
                    "residual",
                ]
            ].head(1000),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# PAGE 3 — MODEL COMPARISON
# =========================================================

elif page == "🤖 Model Comparison":

    st.title("🤖 Model Comparison")

    st.write(
        "Comparison of classical machine-learning models "
        "and the final PyTorch deep-learning model using "
        "earthquake-grouped cross-validation."
    )

    comparison = data["comparison"]

    if comparison is None:

        st.error(
            "final_model_comparison.csv not found."
        )

        st.stop()

    comparison = comparison.copy()

    st.subheader(
        "Grouped 5-Fold Cross-Validation"
    )

    st.caption(
        "Entire earthquake events are kept together during "
        "validation. This evaluates generalization across "
        "unseen earthquake events."
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    mlp_row = comparison[
        comparison["Model"] == "PyTorch MLP"
    ]

    if not mlp_row.empty:

        row = mlp_row.iloc[0]

        st.success(
            f"""
            **Best grouped-CV model: PyTorch MLP**

            RMSE = {row['Grouped CV RMSE']:.4f}

            MAE = {row['Grouped CV MAE']:.4f}

            R² = {row['Grouped CV R2']:.4f}
            """
        )

    st.divider()

    fig = px.bar(
        comparison,
        x="Model",
        y="Grouped CV RMSE",
        title="Grouped CV RMSE",
        text_auto=".4f",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    fig = px.bar(
        comparison,
        x="Model",
        y="Grouped CV R2",
        title="Grouped CV R²",
        text_auto=".4f",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.divider()

    st.subheader(
        "Final Unseen-Earthquake Test"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("RMSE", "0.3113")
    c2.metric("MAE", "0.2800")
    c3.metric("R²", "0.6099")

    st.info(
        """
        These results come from six earthquakes that were
        completely excluded from model development.
        """
    )


# =========================================================
# PAGE 4 — INTERPRETABILITY
# =========================================================

elif page == "🔍 Interpretability":

    st.title(
        "🔍 Model Interpretability"
    )

    st.write(
        """
        Interpretation of the final PyTorch MLP using
        permutation feature importance and one-feature
        response analysis.
        """
    )

    importance = data["importance"]

    if importance is not None:

        st.subheader(
            "Permutation Feature Importance"
        )

        importance = importance.copy()

        if "RMSE_Increase" in importance.columns:

            importance_plot = (
                importance
                .sort_values(
                    "RMSE_Increase",
                    ascending=True,
                )
            )

            fig = px.bar(
                importance_plot,
                x="RMSE_Increase",
                y="Feature",
                orientation="h",
                title=(
                    "Increase in RMSE after "
                    "feature permutation"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.dataframe(
            importance,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    st.subheader(
        "Model Response Analysis"
    )

    response_files = {
        "Depth": data["response_depth"],
        "Hypocentral Distance":
            data["response_distance"],
        "Magnitude":
            data["response_magnitude"],
        "Vs30":
            data["response_vs30"],
    }

    for name, df in response_files.items():

        if df is None or df.empty:
            continue

        st.markdown(
            f"### Response to {name}"
        )

        numeric_columns = [
            col
            for col in df.columns
            if pd.api.types.is_numeric_dtype(
                df[col]
            )
        ]

        if len(numeric_columns) >= 2:

            x_col = numeric_columns[0]
            y_col = numeric_columns[-1]

            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                markers=True,
                title=(
                    f"Predicted response vs {name}"
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

    st.divider()

    st.subheader("Interpretation")

    st.write(
        """
        Permutation importance measures how much predictive
        performance changes when an individual feature is
        disrupted.

        In the final model, event depth is the strongest
        contributor, followed by depth-distance interaction,
        logarithmic distance, and Vs30-related variables.

        This is consistent with the physical expectation that
        earthquake source depth, propagation distance and
        site conditions strongly influence observed ground
        motion.
        """
    )


# =========================================================
# PAGE 5 — PGA PREDICTION
# =========================================================

elif page == "🎯 PGA Prediction":

    st.title(
        "🎯 Ground-Motion Prediction"
    )

    st.write(
        """
        Use the trained PyTorch MLP to generate a ground-motion
        estimate for a new set of earthquake and site parameters.
        """
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        magnitude = st.number_input(
            "Magnitude",
            min_value=3.0,
            max_value=9.5,
            value=5.5,
            step=0.1,
        )

        event_depth = st.number_input(
            "Depth (km)",
            min_value=0.0,
            max_value=500.0,
            value=20.0,
            step=1.0,
        )

        epicentral_distance = st.number_input(
            "Epicentral Distance (km)",
            min_value=0.1,
            max_value=1000.0,
            value=100.0,
            step=1.0,
        )

        hypocentral_distance = st.number_input(
            "Hypocentral Distance (km)",
            min_value=0.1,
            max_value=1000.0,
            value=105.0,
            step=1.0,
        )

    with c2:

        vs30 = st.number_input(
            "Vs30 (m/s)",
            min_value=50.0,
            max_value=2000.0,
            value=500.0,
            step=10.0,
        )

        grid_lat = st.number_input(
            "Grid Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=20.0,
            step=0.1,
        )

        grid_lon = st.number_input(
            "Grid Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=78.0,
            step=0.1,
        )

    st.divider()

    predict_button = st.button(
        "Predict Ground Motion",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        try:

            predicted_log_pga, predicted_pga = (
                predict_log_pga(
                    magnitude=magnitude,
                    event_depth=event_depth,
                    epicentral_distance_km=
                        epicentral_distance,
                    hypocentral_distance_km=
                        hypocentral_distance,
                    vs30=vs30,
                    grid_lat=grid_lat,
                    grid_lon=grid_lon,
                )
            )

            st.success(
                "Prediction generated successfully."
            )

            st.divider()

            st.subheader(
                "Model Prediction"
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Predicted log-PGA",
                f"{predicted_log_pga:.4f}",
            )

            c2.metric(
                "Predicted PGA",
                f"{predicted_pga:.4f}",
            )

            st.divider()

            st.subheader(
                "Input Summary"
            )

            summary = pd.DataFrame(
                {
                    "Parameter": [
                        "Magnitude",
                        "Depth (km)",
                        "Epicentral Distance (km)",
                        "Hypocentral Distance (km)",
                        "Vs30 (m/s)",
                        "Latitude",
                        "Longitude",
                    ],
                    "Value": [
                        magnitude,
                        event_depth,
                        epicentral_distance,
                        hypocentral_distance,
                        vs30,
                        grid_lat,
                        grid_lon,
                    ],
                }
            )

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                """
                This prediction is a machine-learning estimate
                based on the feature distribution used during
                model development. It should not be interpreted
                as an earthquake early-warning system or
                structural-safety assessment.
                """
            )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )