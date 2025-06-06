# FinalRapport Training Pipeline

## Quick start
1. `cd FinalRapport`
2. `python src/train_pipeline.py`  
   or open `notebooks/train_pipeline.ipynb`

## File map
- `src/data_loader.py` – read per-subject CSV files and merge them.
- `src/feature_extraction.py` – windowing helpers and placeholder feature extraction.
- `src/preprocessing.py` – build feature matrix, balance classes.
- `src/train_pipeline.py` – run Stratified K-Fold and LOSO validation then save `lightweight_rf.pkl`.
- `notebooks/train_pipeline.ipynb` – optional notebook interface (empty stub).

## What still needs filling in
- Replace the TODO sections in `feature_extraction.py` with final EMG/IMU features.
- Update `SAMPLE_RATE_HZ` in `train_pipeline.py` to match the recordings.

