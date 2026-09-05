# Interview Guide — Earthquake Ground-Motion ML Project

## 30-second explanation

"I built an end-to-end ground-motion prediction system using roughly 2.87 million ShakeMap observations from 40 earthquakes. I combined earthquake source parameters, propagation geometry, geographic information, and Vs30 site conditions. I compared Elastic Net, a physically engineered Elastic Net, LightGBM, and a PyTorch MLP. The key methodological choice was earthquake-grouped validation so observations from the same earthquake never leak across validation boundaries. The MLP achieved an R² of 0.802 in grouped CV and 0.610 on six completely unseen earthquakes. I then built residual, spatial, and permutation-importance analyses and deployed the model through Streamlit."

## Why not random train/test split?

Because observations from the same earthquake are correlated. A random split could place highly related measurements from one event into both training and validation, creating overly optimistic estimates.

## Why log-PGA?

PGA is highly skewed and spans a wide dynamic range. Modeling log10(PGA) makes the regression target more numerically manageable and better represents multiplicative changes.

## Why Vs30?

Vs30 is a commonly used proxy for near-surface site conditions. It helps represent site-dependent amplification and is therefore physically meaningful for ground-motion prediction.

## Why use both epicentral and hypocentral distance?

They represent different geometric descriptions of source-to-site separation. Hypocentral distance also incorporates depth.

## Why feature engineering?

The model can benefit from physically meaningful transformations such as:

- log distance
- magnitude²
- magnitude × distance
- depth × distance
- log Vs30

These allow nonlinear or interaction effects to be represented explicitly.

## Why compare classical ML with a neural network?

The baselines establish whether deep learning actually adds value. Without them, claiming that the MLP is superior would be weak.

## Why did unseen-event performance drop?

The final test contains earthquake events that the model never saw during development. Their source properties, spatial patterns, site coverage, and residual structure can differ from the development events.

## What is the most important feature?

Permutation analysis identifies event depth as the strongest contributor, followed by depth-distance interaction and distance-related features. Vs30-related variables also contribute substantially.

## Biggest limitation?

The number of independent earthquake events is small relative to the number of individual observations. Millions of observations do not equal millions of independent earthquakes.

## What would you do next?

1. Expand the number of earthquake events.
2. Add regional/geological metadata.
3. Test spatially stratified generalization.
4. Evaluate uncertainty intervals.
5. Compare against established ground-motion equations.
6. Test temporal/geographic out-of-distribution splits.
7. Calibrate predictions for engineering use cases.
