# DISNEYYYY

Simple scikit-learn pipeline that predicts a film's inflation-adjusted box office gross using a handful of metadata features pulled from Disney's historical catalog.

## Quick start
1. Create/activate a virtual environment.
2. `pip install -r requirements.txt`
3. `python train.py`

The training script prints test metrics to the console and writes them to `artifacts/metrics.json` so the run is reproducible.

> Prefer a notebook? Open `submission.ipynb` (already executed) or re-run it to regenerate the same pipeline, plots, and metrics inside Jupyter.

## Data
- `data/disney_movies_total_gross.csv` – movie title, release date, genre, MPAA rating, nominal and inflation-adjusted grosses.  
  The training pipeline parses release dates, converts currency strings to floats, and drops rows without usable targets.

## Pipeline details
- **Features:** release year, title length, genre, and MPAA rating.
- **Target:** `inflation_adjusted_gross` (continuous).
- **Preprocessing:** numeric columns imputed with the median + standardized; categoricals imputed with the most frequent value and one-hot encoded.
- **Model:** `HistGradientBoostingRegressor` wrapped in the preprocessing pipeline.
- **Split:** random 80/20 train/test with a fixed seed (`42`) for repeatability.

## Results
Latest run (`python train.py`) produced:
- Test R²: **0.469**
- Test MAE: **$100.7M**
- Test MAPE: **5.56**

The same figures are captured inside `submission.ipynb` and persisted to `artifacts/metrics.json`. More expressive features (e.g., franchise tags, character info, macro revenue context) or richer models should improve accuracy, but this satisfies the one-hour delivery constraint with a fully reproducible script.
