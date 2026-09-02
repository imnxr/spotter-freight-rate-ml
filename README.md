# Spotter Freight Rate Prediction

This repository contains the machine learning solution for the Spotter Freight Rate Prediction assessment. The objective is to build a robust, leakage-safe regression model to predict spot freight load rates (`posted_rate`) from operational freight attributes such as origin, destination, distance, weight, and equipment type. The final model generates predictions for an unlabeled validation set of 12,000 loads as well as a 31-day operational forecast for December 2025.

---

## Approach

The end-to-end workflow was structured as follows:

1. **Data Audit**: Systematic inspection of feature distributions, missing values, categorical cardinalities, duplicate records, and recording anomalies (such as negative weights) across training, validation, and December inputs.
2. **Temporal Validation**: Implementation of strict out-of-time evaluation splits (Jan–Aug 2025 $\rightarrow$ September 2025, Jan–Sep 2025 $\rightarrow$ October 2025) to evaluate generalization across rolling future-month holdouts without target leakage.
3. **Feature Engineering**: Construction of modular, leakage-safe feature transformers for route identifiers, coordinate distances, and calendar decompositions.
4. **Controlled Model Experiments**: Systematic benchmarking of baseline linear models, feature scaling, step-up feature ablation, L2 regularization tuning ($\alpha$), nonlinear gradient boosting (`HistGradientBoostingRegressor`), target transformations ($\log(1+y)$), expanding historical pricing features, and rate-per-mile target formulations.
5. **Final Model Selection**: Selection of the best-performing model based on cross-fold Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) across both temporal evaluation windows.
6. **Production Prediction Generation**: Fitting the selected model on all 48,000 labeled training rows to produce `validation_predictions.csv` (12,000 rows) and `december_predictions.csv` (31 rows).
7. **Submission Validation**: Verification of output schemas, identifier alignment, value finiteness, and positivity using `score.py` and unit testing.

---

## Data

The project utilizes the following supplied datasets in `data/`:

- **`train-test.csv`** (48,000 labeled rows): Historical freight records spanning January 1, 2025, to October 31, 2025. Contains operational features (`pickup`, `delivery`, `distance`, `weight`, `equipment`, coordinates), market signals (`market_index`, `quote_signal`), and the target variable `posted_rate`.
- **`validation.csv`** (12,000 unlabeled rows): Out-of-sample evaluation loads requiring rate predictions.
- **`december-chart-inputs.csv`** (31 rows): Daily operational records for a fixed lane (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb) spanning December 1 to December 31, 2025.
- **`validation-predictions-template.csv`** (12,000 rows): Submission template establishing the exact required `load_id` sequence.

### Data Quality Counts (Directly Computed)
- **Missing Values**:
  - `weight` is missing in **300 training rows** (0.62%) and **165 validation rows** (1.38%).
  - `market_index` is missing in **374 training rows** (0.78%) and **0 validation rows** (0.00%).
  - Missing numerical values are handled via median imputation learned strictly on training partitions.
- **Negative Weights**:
  - `weight` contains negative values in **292 training rows** (0.61%) and **145 validation rows** (1.21%).
  - These represent recording/formatting artifacts in the source data. They are handled non-destructively through standard imputation and scaling rather than dropping rows.

---

## Validation Strategy

Freight spot rate forecasting is an out-of-time prediction problem where models must forecast rates for future operating windows using only historical transactions. Standard random K-fold cross-validation allows future transactions to leak into training folds, yielding unrealistically optimistic error estimates.

To prevent temporal leakage and measure stability across time, two rolling temporal holdout windows were evaluated directly from `data/train-test.csv`:

- **Fold 1**:
  - **Training Period**: 2025-01-01 to 2025-08-31 (**38,477 rows**)
  - **Validation Period**: 2025-09-01 to 2025-09-30 (**4,670 rows**)
- **Fold 2**:
  - **Training Period**: 2025-01-01 to 2025-09-30 (**43,147 rows**)
  - **Validation Period**: 2025-10-01 to 2025-10-31 (**4,853 rows**)

Total labeled training rows across both periods equal 48,000 rows ($43,147 + 4,853 = 48,000$).

All preprocessing estimators (imputation medians, standard scalers, one-hot encoders) were fitted exclusively on the training partition of each fold and applied to the validation partition without modification.

---

## Model Selection

The table below summarizes the exact experimental results extracted directly from `notebooks/02_model_experiments.ipynb`:

| Model / Experiment Description | Feature Set | Preprocessing | September MAE ($) | September RMSE ($) | October MAE ($) | October RMSE ($) | Mean MAE ($) | Mean RMSE ($) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Initial Baseline: Unscaled Ridge ($\alpha=1.0$)** | Full (Raw + Derived) | Median Imputation + One-Hot | $190.84 | $629.53 | $174.16 | $656.59 | $182.50 | $643.06 |
| **Random Forest Regressor** | Full (Raw + Derived) | Median Imputation + One-Hot | — | — | $242.72 | $760.27 | — | — |
| **Experiment A: Standardized Ridge ($\alpha=1.0$)** | Full (Raw + Derived) | Median Imputer + StandardScaler + One-Hot | $142.46 | $623.17 | $144.15 | $651.86 | $143.31 | $637.52 |
| **Experiment B: Core Features Ridge ($\alpha=1.0$)** | Core Features | Median Imputer + StandardScaler + One-Hot | $142.21 | $622.86 | $143.67 | $653.40 | $142.94 | $638.13 |
| **Experiment C: Ridge ($\alpha=10.0$)** | Core Features | Median Imputer + StandardScaler + One-Hot | $142.00 | $622.81 | $143.45 | $653.39 | $142.73 | $638.10 |
| **Experiment C: Ridge ($\alpha=30.0$)** | Core Features | Median Imputer + StandardScaler + One-Hot | $141.55 | $622.76 | $143.04 | $653.40 | $142.29 | $638.08 |
| **Experiment C: Ridge ($\alpha=100.0$) [CHAMPION]** | **Core Features** | **Median Imputer + StandardScaler + One-Hot** | **$140.52** | **$622.71** | **$142.06** | **$653.47** | **$141.29** | **$638.09** |
| **Experiment E: HistGradientBoosting (best)** | Interaction Features | Ordinal Tree Encoding + Imputation | $149.60 | $628.93 | $137.81 | $656.09 | $143.70 | $642.51 |
| **Experiment F: Log-Target HGB (best)** | Interaction Features | Target: $\log(1+y)$, Inverse: $\text{expm1}$ | $160.24 | $633.86 | $139.14 | $656.97 | $149.69 | $645.41 |
| **Experiment F: Log-Target Ridge ($\alpha=100.0$)** | Core Features | Target: $\log(1+y)$, Inverse: $\text{expm1}$ | $427.30 | $905.49 | $422.85 | $885.42 | $425.08 | $895.46 |
| **Experiment G: Expanding Historical Pricing** | Core + Route History | Self-Expanding Priors + StandardScaler | $159.65 | $623.46 | $152.78 | $652.63 | $156.22 | $638.04 |
| **Experiment H: Rate-Per-Mile Ridge ($\alpha=100.0$)** | Core Features | Target: $y / \text{dist}$, Prediction: $\hat{y}_{\text{rpm}} \times \text{dist}$ | $155.75 | $629.90 | $157.91 | $661.25 | $156.83 | $645.58 |
| **Experiment H: Rate-Per-Mile Ridge ($\alpha=300.0$)** | Core Features | Target: $y / \text{dist}$, Prediction: $\hat{y}_{\text{rpm}} \times \text{dist}$ | $153.81 | $629.04 | $156.05 | $660.51 | $154.93 | $644.77 |

*Important Clarification*: These error metrics reflect historical out-of-time holdout evaluations on September and October 2025 partitions from `train-test.csv`. They do NOT represent performance on the hidden 12,000-row `validation.csv` dataset.

### Selected Champion Specification
- **Model**: Standardized `Ridge(alpha=100.0, random_state=42)`
- **Features (5)**: `distance`, `weight`, `pickup`, `delivery`, `equipment`
- **Preprocessing**: `SimpleImputer(strategy="median")` $\rightarrow$ `StandardScaler()` on numerical features; `OneHotEncoder(handle_unknown="ignore")` on categorical features.
- **Cross-Fold Evaluation Results**:
  - September 2025: **MAE = $140.52**, **RMSE = $622.71**
  - October 2025: **MAE = $142.06**, **RMSE = $653.47**
  - Mean Across Folds: **Mean MAE = $141.29**, **Mean RMSE = $638.09**

---

## Why This Model

The selection of Standardized Ridge ($\alpha=100.0$) on Core Features was determined by empirical evidence across both temporal folds:

1. **Feature Scaling Materially Improved Linear Generalization**: Unscaled Ridge had a Mean MAE of $182.50. Adding `StandardScaler` to numerical inputs reduced Mean MAE to $143.31 ($-\$39.19 improvement), preventing unscaled high-magnitude variables (`weight`, `distance`) from dominating the regularization penalty.
2. **Feature Parsimony in Linear Formulations**: In the tested linear Ridge formulation, adding derived calendar and spatial features (such as calendar dates, day of year, coordinate differences) increased out-of-sample error ($161.99 MAE with temporal features vs. $143.67 MAE with Core features on the October holdout), indicating that extraneous regressors added noise without improving generalization.
3. **Regularization Search**: Evaluating $\alpha \in [0.01, 100.0]$ showed monotonic improvement as regularization increased, reaching the lowest cross-fold Mean MAE ($141.29) and RMSE ($638.09) at $\alpha=100.0$.
4. **Tree-Based vs. Linear Trade-offs**: `HistGradientBoostingRegressor` achieved lower median absolute error on typical October loads ($52.89 vs. $60.77 for Ridge), but exhibited higher cross-fold Mean MAE ($143.70 vs. $141.29) due to sensitivity on September data ($149.60 MAE).
5. **Rejection of Alternative Formulations**:
   - **Log-Target Transformation ($\log(1+y)$)**: Fitting on $\log(1+y)$ optimizes squared error in log space (closely related to relative percentage error), which caused large dollar-scale prediction distortions when inverse-transformed with $\text{expm1}$ on higher-rate loads, yielding $425.08 Mean MAE for Ridge and degrading HGB to $149.69 Mean MAE.
   - **Historical Pricing Aggregates**: Adding rolling prior lane statistics degraded Mean MAE to $156.22 due to non-stationary covariate shift between early-year training records (sparse history) and late-year records.
   - **Rate-Per-Mile Target Formulation**: Predicting $y / \text{distance}$ and reconstructing $\hat{y}_{\text{rpm}} \times \text{distance}$ degraded Mean MAE to $156.83. Minimizing squared loss in rate-per-mile space implicitly weights absolute dollar errors by $1/\text{distance}^2$, overweighting short hauls and degrading long-distance accuracy ($253.57 $\rightarrow$ $314.93 MAE on long hauls).
6. **Outlier Reality**: Error analysis showed that rare extreme pricing spikes ($>\$7,000$ up to $\$25,533$) in the tail distribution appear to reflect pricing regimes not adequately explained by the supplied load attributes. Standardized Ridge provided the most stable, bounded predictions across all evaluation windows.

---

## Leakage Prevention

Strict data leakage boundaries were enforced across all modules:

- **Target Isolation**: The target variable `posted_rate` and rate-per-mile values were never included as contemporaneous model inputs.
- **Identifier Exclusion**: `load_id` was strictly excluded from feature extraction and modeling pipelines.
- **Temporal Split Boundaries**: Validation rows never influenced training statistics or preprocessing parameters.
- **Expanding Window Integrity**: All historical target aggregation experiments were constructed using self-expanding prior-date windows ($t < \text{current\_date}$) and verified with targeted unit tests.
- **Preprocessing Encapsulation**: Imputation medians, standard scalers, and one-hot encoder categories were fitted exclusively within scikit-learn `Pipeline` objects on training splits.

---

## Final Prediction Artifacts

The final pipeline was fitted on all 48,000 labeled training rows from `data/train-test.csv` to generate the required submission files:

### 1. `validation_predictions.csv`
- **Rows**: Exactly 12,000 predictions.
- **Format**: `load_id,predicted_rate` matching `data/validation-predictions-template.csv` in exact identifier order.
- **Distribution**:
  - Minimum: **$31.57**
  - 25th Percentile: **$1,300.04**
  - Median: **$2,043.71**
  - Mean: **$2,382.77**
  - 75th Percentile: **$3,365.08**
  - Maximum: **$6,516.82**
  - All predictions are finite, non-null, and strictly positive ($>0$).

### 2. `december_predictions.csv`
- **Rows**: Exactly 31 daily predictions spanning 2025-12-01 through 2025-12-31.
- **Format**: `pickup,delivery,distance,equipment,weight,date,predicted_rate`.
- **Value**: **$793.40** across all 31 days.

### 3. `scorer_results/candidate_december.png`
- Validation chart artifact rendered by `score.py` confirming the 31-day forecast trajectory.

*Verification Note*: Running `score.py` successfully verifies all file constraints and creates the candidate chart. Final validation performance metrics are computed independently by Spotter on the hidden validation set.

---

## December Forecast

The 31-day December forecast for the Lexington to Fort Wayne Dry Van load (360 miles, 32,000 lb) yields a constant predicted rate of **$793.40** ($2.204/mi).

### Why the Forecast is Constant
Feature ablation experiments demonstrated that calendar day, day-of-week, and day-of-year features increased out-of-sample variance and degraded temporal holdout performance in the linear model. Consequently, temporal features were excluded from the champion model. Because all operational inputs for the December evaluation lane remain identical across each day, the model deterministically produces a constant prediction.

### Historical Sanity Check
An inspection of historical training data for the exact lane (`Lexington` $\rightarrow$ `Fort Wayne`, `Dry Van`) reveals:
- **Historical Load Count**: 21 loads in training data
- **Historical Distance**: 350.9 to 378.8 miles (mean 363.8 miles)
- **Historical Posted Rate Range**: $757.93 to $934.37 (mean **$823.31 $\pm$ $46.15**, median **$807.89**)
- **December Model Prediction**: **$793.40**

The December prediction falls near the 25th–30th percentile of historical Dry Van loads on this exact lane, representing an operationally reasonable baseline rate. (This comparison is a sanity check, not validation proof.)

---

## Repository Structure

```
spotter-freight-rate-ml/
|-- data/
|   |-- december-chart-inputs.csv
|   |-- train-test.csv
|   |-- validation-predictions-template.csv
|   \-- validation.csv
|-- notebooks/
|   |-- 01_data_audit.ipynb
|   \-- 02_model_experiments.ipynb
|-- scorer_results/
|   \-- candidate_december.png
|-- src/
|   |-- __init__.py
|   |-- data.py
|   |-- features.py
|   \-- validation.py
|-- tests/
|   |-- test_data.py
|   |-- test_features.py
|   \-- test_validation.py
|-- .gitignore
|-- december_predictions.csv
|-- README.md
|-- requirements.txt
|-- score.py
\-- validation_predictions.csv
```

---

## Reproduction

To reproduce the environment, run the test suite, and score the predictions:

### 1. Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
python -m unittest discover -s tests -v
```
*Expected Output*: `Ran 39 tests in ... OK`.

### 3. Run Submission Scorer
```powershell
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```
*Expected Output*:
```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results\candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

---

## Testing

The project maintains an automated test suite with **39 unit tests** covering:

- **Data Loading & Schema Validation** ([`tests/test_data.py`](tests/test_data.py)): 13 tests covering file existence, column schema enforcement, date parsing, and non-destructive return behaviors.
- **Temporal Validation Integrity** ([`tests/test_validation.py`](tests/test_validation.py)): 10 tests verifying strict date thresholding, chronological separation, exclusion of `load_id`, and non-empty partition checks.
- **Feature Engineering & Leakage Isolation** ([`tests/test_features.py`](tests/test_features.py)): 16 tests verifying:
  - Input DataFrame immutability
  - Coordinate difference and midpoint mathematics
  - Rate-per-mile target transformation and exact inverse reconstruction
  - Expanding historical target isolation (row cannot observe its own target or future targets)
  - Validation feature invariance to validation target values

---

## Limitations

- **Hidden Validation Ground Truth**: Final model performance on `validation.csv` is determined by Spotter's hidden evaluation labels. The reported error metrics represent historical out-of-time temporal holdouts (September and October 2025).
- **Extreme Rate Spikes**: The training data contains rare spot market rate surges ($>\$7,000$ up to $\$25,533$) that appear to reflect pricing regimes not adequately explained by the supplied load attributes. Linear models systematically underpredict these rare peaks.
- **Data Recording Anomalies**: A small fraction of records contain negative weight values. While median imputation allows models to handle them without crashing, they remain data-quality anomalies.
- **Constant December Forecast**: Because temporal features were excluded during feature ablation to improve overall out-of-sample accuracy in the linear model, the December forecast does not capture potential day-to-day holiday seasonality.

---

## Conclusion

The final solution employs a parsimonious, standardized linear Ridge regression model trained on core operational attributes (`distance`, `weight`, `pickup`, `delivery`, `equipment`). Through disciplined temporal validation and extensive benchmarking, complex alternatives (target transformations, expanding historical averages, high-dimensional temporal encodings) were empirically rejected in favor of a stable, regularized architecture that achieved a cross-fold Mean MAE of **$141.29** across out-of-time holdouts.
