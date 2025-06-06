<!-- README for FinalRapport1 (Final Report 2) -->
# FinalRapport2 Training Guide

This folder contains the skeleton code for training a RandomForest model on your EMG/IMU dataset. Several sections still require implementation.

## What still needs to be filled in
1. **Feature extraction** - Edit `src/feature_extraction.py` and replace the bodies of
   `extract_emg_features` and `extract_imu_features` with your real feature
   calculations (e.g. RMS, MAV, zero crossing for EMG and mean, variance or orientation features for IMU).
2. **Sample rate** - Open `src/train_pipeline.py` and set `SAMPLE_RATE_HZ` to the
   acquisition frequency used in your recordings.

## Preparing the data
1. Place all subject CSV files in `FinalRapport1/data/`.
   Each CSV should contain your sensor channels plus a `label` column. The file
   name (without `.csv`) will be used as the subject identifier.
2. Typical columns might look like:
   `emg1, emg2, ..., imu_ax, imu_ay, imu_az, label`.

## Training step by step
1. From the repository root run:
   ```bash
   python FinalRapport1/src/train_pipeline.py
   ```
2. The script loads all CSVs, windows the data and extracts the features you
   implemented.
3. Stratified 5-fold and leave-one-subject-out validation are performed.
4. After training, `lightweight_rf.pkl` and `confusion_matrix.png` are saved in
   `FinalRapport1/`.

Once the feature extraction functions are implemented and your data is placed in
`data/`, running the above command will train your final model.
