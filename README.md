# MLB Pitcher Strikeout Rate Projection

An explainable, team-agnostic machine-learning workflow for projecting Major League Baseball pitcher strikeout percentage (K%). The repository uses the 2025 season as a case study, but the modeling design is applicable to any club: predictions are generated for every player in the input population and team identity is excluded from the model.

## Project overview

The model predicts 2025 K% for 873 players using only information that would have been available before the 2025 regular season. Target-season K%, batters faced (TBF), Stuff+, and team are excluded from training and feature construction.

The final estimator is a Ridge regression selected through time-ordered validation. It combines a transparent Marcel-style projection with K% history, workload, availability, Stuff+, age, and missing-history indicators. This approach keeps the model interpretable while improving equal-player accuracy over the standalone benchmarks.

Across 1,718 chronological 2023-2024 validation pitcher-seasons, Ridge achieved:

- 5.31 percentage-point MAE, versus 5.85 for Marcel and 6.17 for prior-season K%
- 7.48 percentage-point RMSE
- 3.60 percentage-point TBF-weighted MAE

On the untouched 2024 validation season, Ridge achieved a 5.26-point MAE and 7.26-point RMSE.

## Why the model is team-agnostic

- It scores the full player population rather than a single organization's roster.
- Team identity is never used as a feature, avoiding team-specific effects and accidental target-season leakage.
- Feature engineering depends only on player history, league context, age, workload, and pitch-quality information available before the forecast season.
- The validation design mirrors a real preseason forecast by always training on earlier seasons and predicting a later one.

## Validation design

Random train/test splits are inappropriate for forecasting future performance because they can allow later-season information to influence an earlier prediction. The project therefore uses expanding, chronological folds.

| Forecast stage | Information used | Purpose |
|---|---|---|
| Build 2022 training rows | 2021 history creates features; actual 2022 K% supplies the outcome | First learnable feature/outcome examples |
| Predict 2023 | Train on 2022 target rows; build player features through 2022 | Select the Ridge penalty and feature group |
| Predict 2024 | Refit on 2022-2023 target rows; build player features through 2023 | Untouched final validation |
| Predict 2025 | Refit the frozen model on 2022-2024 target rows; build features through 2024 | Produce the case-study projections |

Because the dataset begins in 2021, that season can provide lagged features but cannot itself be a prediction target.

## Model inputs

The selected 20-feature specification includes:

- **K% history:** prior-season, most recent observed, recent two-year, available-history, and Marcel-style K%
- **Workload and availability:** prior, recent, and available-history TBF; time since last observed; missing-history indicators
- **Pitch quality:** prior, most recent, and recency-weighted Stuff+, plus Stuff+ change and missingness indicators
- **Age:** age centered at 30 and its square, allowing a nonlinear relationship without assuming age 30 is the performance peak

The model uses `log(1 + TBF)` because workload is highly right-skewed. Missing continuous values are imputed, and Ridge inputs are standardized using parameters learned only from each training fold.

## Model comparison

Errors are calculated on the same 2023-2024 validation player-seasons and expressed in K percentage points.

| Model | MAE | RMSE | TBF-weighted MAE |
|---|---:|---:|---:|
| Ridge | 5.31 | 7.48 | 3.60 |
| OLS | 5.31 | 7.48 | 3.60 |
| Gradient boosting | 5.46 | 7.58 | 3.71 |
| Marcel | 5.85 | 8.46 | 3.33 |
| Prior-season K% | 6.17 | 8.87 | 3.91 |

Unweighted MAE is the primary metric because the goal is one prediction per player. Marcel has the best TBF-weighted MAE because that metric gives established, high-workload pitchers more influence. Ridge performs better across the larger population of players with limited or missing history.

## Repository structure

```text
.
|-- data/
|   `-- k_2026.csv                 # Player-season modeling dataset
|-- notebooks/
|   |-- k_projection.ipynb         # Executable analysis and model workflow
|   |-- k_projection.html          # Browser-friendly rendered notebook
|   `-- k_projection.pdf           # PDF report
|-- outputs/
|   `-- k_2025_predictions.csv     # Final 2025 K% projections
|-- environment.yml                # Reproducible Conda environment
`-- README.md
```

## Reproduction

From the project directory:

```bash
conda env create -f environment.yml
conda activate mlb-k-projection
jupyter lab
```

Open `notebooks/k_projection.ipynb` and run all cells from top to bottom. The notebook locates the project root whether Jupyter starts from the root or `notebooks/`, rebuilds every historical fold, refits the final model, validates all 873 predictions, and writes `outputs/k_2025_predictions.csv`.

The prediction file uses decimal K% (`0.250000` means 25.00%).

## Limitations and next steps

- The dataset begins in 2021, leaving only 2023 for model selection and 2024 for untouched final validation.
- Available-history statistics are not complete-career measures.
- Starter/reliever role is not provided; TBF alone cannot reliably distinguish role, injury, or midseason promotion.
- Position players making emergency pitching appearances can be overprojected when missing history is regressed toward the pitcher league average.
- The model does not directly include injury status, expected workload, pitch mix, velocity, handedness, park, catcher, or opponent context.
- A strong next experiment would add preseason-available swinging-strike rate, then test whether it improves out-of-time accuracy beyond K% history and Stuff+.

## Data provenance and publication note

The included player-season dataset is the modeling input for this case study. Before redistributing this repository publicly, document the dataset's original source and confirm that its terms allow redistribution. If redistribution is not permitted, remove the tracked data and generated player-level predictions, publish a schema/sample instead, and add retrieval instructions.

## References and assistance

- [Marcel projection framework](https://tangotiger.net/marcel/)
- [FanGraphs Stuff+, Location+, and Pitching+ primer](https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/)
- [FanGraphs plate-discipline and SwStr% definitions](https://library.fangraphs.com/pitching/plate-discipline-o-swing-z-swing-etc/)
- [scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html)
- [scikit-learn histogram gradient boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
- [scikit-learn imputation](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [scikit-learn scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [OpenAI Codex](https://openai.com/codex/) assisted with code structure, review, statistical explanations, and documentation. All model logic and reported outputs are reproducible in the notebook.
