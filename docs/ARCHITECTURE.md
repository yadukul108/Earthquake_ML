# System Architecture

```mermaid
flowchart TD
    A[Earthquake Metadata] --> D[Data Integration]
    B[ShakeMap Ground Motion] --> D
    C[Vs30 / Site Conditions] --> D

    D --> E[Quality Control]
    E --> F[EDA + Physical Analysis]
    F --> G[Feature Engineering]

    G --> H1[Elastic Net]
    G --> H2[Physical Elastic Net]
    G --> H3[LightGBM]
    G --> H4[PyTorch MLP]

    H1 --> I[Earthquake-Grouped CV]
    H2 --> I
    H3 --> I
    H4 --> I

    I --> J[Model Selection]
    J --> K[Unseen Earthquake Test]

    K --> L[Metrics]
    K --> M[Residual Analysis]
    K --> N[Spatial Analysis]
    K --> O[Interpretability]

    J --> P[Saved Model Artifacts]
    P --> Q[Streamlit Inference App]

    L --> R[Streamlit Dashboard]
    M --> R
    N --> R
    O --> R
    Q --> R
```

## Core Principle

The split is performed at the **earthquake-event level** rather than the observation level.

This reduces event-level leakage and makes the final evaluation substantially more representative of deployment on a new earthquake.
