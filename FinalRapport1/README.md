<!-- README for FinalRapport1 -->
# FinalRapport1 Training Pipeline

## Quick start
Run from repo root:
```bash
python FinalRapport1/src/train_pipeline.py
```
This loads all CSVs from `FinalRapport1/data/`, trains a RandomForest and saves
`lightweight_rf.pkl` back in this folder.

## File map
- `src/data_loader.py` – merge subject CSV files.
- `src/feature_extraction.py` – window helpers and stub feature extraction.
- `src/preprocessing.py` – build dataset and balance classes.
- `src/train_pipeline.py` – run 5-Fold + LOSO validation, save model.
- `notebooks/train_pipeline.ipynb` – optional notebook wrapper.

## TODO
- Replace stubbed feature extractors in `feature_extraction.py`.
- Set `SAMPLE_RATE_HZ` in `train_pipeline.py`.
