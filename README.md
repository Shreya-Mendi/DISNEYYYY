# Disney Character Popularity Mini-Project
We used ChatGPT 5 on 11/11 at 12:45pm to generate some code and content for this project.

This repo contains a compact machine-learning pipeline that predicts whether a Disney hero headlines a blockbuster (above-median inflation-adjusted gross) by blending:
- Character descriptors (`hero`, `villain`, signature `song`) from `data/disney-characters.csv`.
- Movie context (`genre`, `MPAA_rating`, `release_year`, grosses) from `data/disney_movies_total_gross.csv`.
- Director metadata from `data/disney-director.csv` plus engineered cues (hero/villain archetypes, release decade/month, song stats).

Random forest + one-hot features keep the model fun while still validating on held-out data.

## Key artifacts
- `notebooks/disney_character_popularity.ipynb`: End-to-end workflow with markdown explanations, preprocessing, model tuning, and validation outputs.
- `requirements.txt`: Minimal stack (NumPy, pandas, scikit-learn, seaborn, matplotlib) needed to rerun the notebook.

## Reproducing the results
1. (Optional) Create/activate a virtual environment.
2. `pip install -r requirements.txt`
3. Launch Jupyter and run `notebooks/disney_character_popularity.ipynb` (or execute headlessly via `python -m nbclient --execute --inplace notebooks/disney_character_popularity.ipynb`).

The notebook will:
1. Merge the character + movie tables and engineer the `is_blockbuster` target.
2. Split the data with stratification and wrap preprocessing/modeling in a single scikit-learn pipeline.
3. Grid-search a `RandomForestClassifier` over depth, estimators, and min-samples-split.
4. Report classification metrics, a confusion matrix, and the most important hero/genre/song signals.

## Model performance (held-out test set)
- Test accuracy: **0.90**
- Test F1 (blockbuster class): **0.91**
- Best model: `RandomForestClassifier` (`n_estimators=800`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=2`)

These scores come from running the executed notebook included in this repo. Feel free to tweak the grid or feature set (e.g., add voice actors or director data) to explore different creative signals before pushing to GitHub and sharing the link.
