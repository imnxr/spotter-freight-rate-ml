# Spotter Freight Rate Prediction

Machine learning solution for the Spotter freight rate prediction assessment. The goal was to predict `posted_rate` for future loads from the information available on each load, then produce the required validation and December prediction files.

The development data contains 48,000 labeled loads from January through October 2025. The supplied validation set contains 12,000 unlabeled loads.

## What I did

I treated the problem as a time-based forecasting task. Instead of randomly splitting the data, I trained on earlier months and tested on a later month. I used the following workflow:

1. Inspect the supplied data and look for missing or unusual values.
2. Build two out-of-time validation folds.
3. Start with a linear Ridge model.
4. Test scaling and feature groups.
5. Tune the Ridge regularization strength.
6. Compare the result with a tree-based model and a few target formulations.
7. Refit the selected model on all labeled data.
8. Generate the required prediction files and run the supplied scorer.

## Data

The assessment supplied four data files:

| File | Rows | Purpose |
| --- | ---: | --- |
| `train-test.csv` | 48,000 | Labeled development data from Jan 1 to Oct 31, 2025 |
| `validation.csv` | 12,000 | Unlabeled loads for the final prediction file |
| `december-chart-inputs.csv` | 31 | Fixed December scenario used by the supplied scorer |
| `validation-predictions-template.csv` | 12,000 | Required prediction IDs and output order |

The labeled data contains pickup and delivery cities, coordinates, distance, equipment, weight, date, `market_index`, `quote_signal`, and the target `posted_rate`.

### Data quality

A small number of records have missing or unusual values.

- `weight` is missing in 300 training rows and 165 validation rows.
- `market_index` is missing in 374 training rows. It is complete in the supplied validation file.
- Negative weights occur in 292 training rows and 145 validation rows.
- No non-positive `posted_rate` values were found in the development data used for modeling.

I kept the rows rather than dropping them. The final model handles missing numeric values with median imputation inside the model pipeline.

## Validation strategy

The validation scheme is deliberately chronological because the final prediction set represents later loads. A random split would allow records from later periods to appear in the training side of a fold.

### Fold 1

Training: January 1 to August 31, 2025, 38,477 rows

Validation: September 1 to September 30, 2025, 4,670 rows

### Fold 2

Training: January 1 to September 30, 2025, 43,147 rows

Validation: October 1 to October 31, 2025, 4,853 rows

For each fold, the imputer, scaler, and categorical encoder were fit only on the training portion. The validation portion was then passed through the fitted pipeline.

## Feature choices

I started with five fields that are available in every final prediction input:

- `distance`
- `weight`
- `pickup`
- `delivery`
- `equipment`

I also tested coordinates, calendar fields, `market_index`, `quote_signal`, route features, and several derived values. In the Ridge experiments, the smaller feature set was more consistent across the two time folds.

One practical benefit is that the final five fields are also present in `december-chart-inputs.csv`, so the same model can produce both required outputs.

## Model selection

The main development results are below. These numbers are from the September and October holdouts, not from the hidden Spotter validation labels.

| Model | September MAE | October MAE | Mean MAE |
| --- | ---: | ---: | ---: |
| Unscaled Ridge, alpha=1 | USD 190.84 | USD 174.16 | USD 182.50 |
| Standardized Ridge, full feature set | USD 142.46 | USD 144.15 | USD 143.31 |
| Core Ridge, alpha=1 | USD 142.21 | USD 143.67 | USD 142.94 |
| Core Ridge, alpha=10 | USD 142.00 | USD 143.45 | USD 142.73 |
| Core Ridge, alpha=30 | USD 141.55 | USD 143.04 | USD 142.29 |
| **Core Ridge, alpha=100** | **USD 140.52** | **USD 142.06** | **USD 141.29** |
| HistGradientBoosting | USD 149.60 | USD 137.81 | USD 143.70 |
| Log-target HGB | USD 160.24 | USD 139.14 | USD 149.69 |
| Log-target Ridge, alpha=100 | USD 427.30 | USD 422.85 | USD 425.08 |
| Expanding historical pricing | USD 159.65 | USD 152.78 | USD 156.22 |
| Rate-per-mile Ridge, alpha=100 | USD 155.75 | USD 157.91 | USD 156.83 |
| Rate-per-mile Ridge, alpha=300 | USD 153.81 | USD 156.05 | USD 154.93 |

The selected model is Ridge regression with `alpha=100`.

### Final model

**Model:** `Ridge(alpha=100, random_state=42)`

**Numeric features:** `distance`, `weight`

**Categorical features:** `pickup`, `delivery`, `equipment`

**Numeric preprocessing:** median imputation, then `StandardScaler`

**Categorical preprocessing:** `OneHotEncoder(handle_unknown="ignore")`

### Development performance

| Holdout | MAE | RMSE |
| --- | ---: | ---: |
| September 2025 | USD 140.52 | USD 622.71 |
| October 2025 | USD 142.06 | USD 653.47 |
| Mean across folds | **USD 141.29** | **USD 638.09** |

The cross-fold mean MAE of USD 141.29 is the main model-development result in this repository. It is not the final hidden validation score.

## Why Ridge won

The biggest improvement came from scaling the numeric features. The initial unscaled Ridge model had a mean MAE of USD 182.50. The standardized version reduced that to USD 143.31.

After that, I compared the core variables with larger feature sets. In the tested Ridge setup, adding calendar and spatial features increased the October holdout MAE instead of improving it.

I then tuned the Ridge penalty over the tested alpha range. The best result was at `alpha=100`, with a small but consistent improvement over lower values on both folds.

I also tested several alternatives:

- HistGradientBoosting had better median error on the October holdout, but its September result was weaker and its mean MAE was higher.
- Training on `log1p(posted_rate)` changed the error profile but made the overall dollar error worse.
- Historical lane pricing features increased mean MAE rather than improving it.
- Predicting `posted_rate / distance` and converting back to dollars also performed worse, especially on long hauls.

For that reason I kept the simpler Ridge model rather than adding complexity without a measurable gain.

## Error analysis

The remaining weakness is the high end of the rate distribution. On the October holdout:

- Median absolute error: USD 60.77
- 90th percentile absolute error: USD 201.91
- 95th percentile absolute error: USD 282.61
- 99th percentile absolute error: USD 1,813.70
- Maximum absolute error: USD 13,884.22

The largest errors were generally underpredictions on unusually expensive long-haul loads. Their supplied load attributes look similar to ordinary loads, so the model has limited information with which to recognize those spikes.

Negative weights are another data-quality issue. They were associated with higher error, but they did not account for the largest October errors.

## Leakage controls

The final model does not use the target as an input.

- `posted_rate` is the target only.
- `load_id` is excluded from the feature set.
- Current-row rate-per-mile is never used as a feature.
- Time-based holdouts keep later labeled rows out of earlier training folds.
- Imputation, scaling, and category encoding are fit on the training side of each fold.
- Historical target features were built with prior-date filtering and were rejected from the final model after testing.

## Final prediction files

After selecting the model, I fit it on all 48,000 labeled rows and generated the required outputs.

### `validation_predictions.csv`

- 12,000 rows
- Columns: `load_id,predicted_rate`
- IDs match the supplied template order
- All predicted values are finite and strictly positive

Prediction distribution:

| Statistic | Value |
| --- | ---: |
| Minimum | USD 31.57 |
| 25th percentile | USD 1,300.04 |
| Median | USD 2,043.71 |
| Mean | USD 2,382.77 |
| 75th percentile | USD 3,365.08 |
| Maximum | USD 6,516.82 |

### `december_predictions.csv`

The December file contains all 31 required dates and the required seven columns. The model gives the fixed scenario a prediction of **USD 793.40** for every day.

### `scorer_results/candidate_december.png`

This is the chart produced by the supplied `score.py` script from the December predictions.

The supplied scorer accepted both prediction files and confirmed the required row counts and value constraints.

## December check

The December input repeats the same operational values for Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. The selected model does not use the date field, so the predicted rate is constant across the month.

As a sanity check only, the training data contains 21 historical loads on this exact lane with Dry Van equipment. Their posted rates range from USD 757.93 to USD 934.37, with a median of USD 807.89. The model's USD 793.40 prediction is inside that observed range.

This is a plausibility check, not evidence about the hidden December labels.

## Repository layout

```text
spotter-freight-rate-ml/
├── data/
│   ├── december-chart-inputs.csv
│   ├── train-test.csv
│   ├── validation-predictions-template.csv
│   └── validation.csv
├── notebooks/
│   ├── 01_data_audit.ipynb
│   └── 02_model_experiments.ipynb
├── scorer_results/
│   └── candidate_december.png
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   └── validation.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_validation.py
├── .gitignore
├── december_predictions.csv
├── README.md
├── requirements.txt
├── score.py
└── validation_predictions.csv
```

## Reproduction

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

Expected result: 39 tests pass.

Run the supplied scorer against the generated prediction files:

```powershell
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The scorer should confirm 12,000 validation predictions, 31 December rows, and create `scorer_results/candidate_december.png`.

## Testing

There are 39 unit tests covering the data loaders, temporal split logic, feature utilities, and leakage checks.

The tests include checks that historical target features cannot see a row's own target or later targets, and that validation features are unchanged by validation target values.

## Limitations

The hidden validation labels are not available during development, so the final Spotter score cannot be reported in advance.

The supplied features do not explain every high-rate spike. That is the main source of large residuals in the time-based holdouts.

Negative weights remain source-data anomalies. The pipeline can process them, but the underlying records should still be checked upstream.

The December forecast is flat because the selected model does not use date. Adding a seasonal effect simply to make the line move would not be justified by the holdout results.

## Conclusion

The final submission uses a standardized Ridge model with `alpha=100` and five operational fields. It achieved a cross-fold mean MAE of **USD 141.29** on the September and October temporal holdouts.

I kept the model deliberately small because the tested alternatives did not produce a better and more consistent result. The final prediction files passed the supplied scorer checks, and the repository includes the notebooks and tests used to reach the final model.