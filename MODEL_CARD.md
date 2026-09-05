# Model Card — Earthquake Ground-Motion MLP

## Model

**Name:** PGA MLP  
**Task:** Regression of log10(PGA)  
**Framework:** PyTorch  
**Input features:** 13  
**Output:** one continuous log-PGA value

## Architecture

```text
13 → Linear(64) → ReLU → BatchNorm
   → Linear(32) → ReLU → BatchNorm
   → Linear(16) → ReLU
   → Linear(1)
```

## Development Evaluation

Grouped 5-fold cross-validation:

| Metric | Mean |
|---|---:|
| RMSE | 0.212796 |
| MAE | 0.177651 |
| R² | 0.801543 |

## Final Unseen-Event Evaluation

| Metric | Result |
|---|---:|
| RMSE | 0.311307 |
| MAE | 0.280031 |
| R² | 0.609882 |

## Intended Use

- research and educational analysis
- portfolio demonstration
- ML experimentation
- exploration of ground-motion patterns

## Out-of-Scope Use

- earthquake early warning
- structural safety decisions
- emergency response decisions
- engineering design without independent validation
- operational seismic hazard assessment

## Known Risks

The model can perform differently on earthquake events whose source properties, geographic distribution, site conditions, or propagation characteristics differ from the development distribution.

The final six-event test demonstrates this explicitly.

## Interpretability

The project includes permutation feature importance and response analysis. These are diagnostic tools and should not be interpreted as causal inference.

## Reproducibility

Inference requires the saved:

- `pga_mlp_development.pt`
- `pga_scaler.pkl`
- `pga_imputer.pkl`

The scaler and imputer must be loaded with `joblib.load()`.
