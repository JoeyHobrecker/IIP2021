# Bicep-Wearable EMG+IMU Pipeline
This repository contains code to collect, preprocess, extract features, and train ML models (with K-Fold and LOSO validation) using an ESP32 wearable that measures EMG and IMU signals during exercise.

## Contents
```
├── /src
│   ├ esp32_sketch.ino
│   ├ pc_data_logger.py
│   ├ data_loader.py
│   ├ preprocessing.py
│   ├ feature_extraction.py
│   └ train_pipeline.py
└── /notebooks
    └ train_pipeline.ipynb
```

## Wiring & Hardware Setup
1. **EMG front-end (ADS1299)**
   - Connect EMG module DATA_OUT pin to **GPIO34** on the ESP32.
   - Power (VCC) → 3.3 V, GND → GND.
2. **MPU-6050 IMU**
   - SDA → **GPIO21**, SCL → **GPIO22** on the ESP32.
   - VCC → 3.3 V, GND → GND.
3. **(Optional) Haptic Feedback Driver**
   - Driver input (+) → **GPIO25**
   - Driver ground (–) → GND
4. Connect ESP32 to PC over USB (for Serial and code upload).

## Dependencies & Installation
- On the ESP32 side:
  1. Install the ESP32 board package in Arduino IDE (or PlatformIO).
  2. Install the ADS1299 library via Library Manager (e.g., "ADS1299" by Texas Instruments or a community fork).
  3. In `esp32_sketch.ino`, verify that `Wire.begin(21, 22)` is used for I²C.
- On the PC side (Python scripts):
```bash
python3 -m venv venv
source venv/bin/activate       # (or `venv\Scripts\activate` on Windows)
pip install --upgrade pip
pip install pandas numpy scikit-learn pyserial matplotlib jupyter
```

## How to Run
1. **Upload firmware to ESP32**
   - Open `/src/esp32_sketch.ino` in Arduino IDE.
   - Select "ESP32 Dev Module," COM port, and 115200 flash speed.
   - Upload the sketch.
   - Press ESP32 reset button.
   - Open Serial Monitor at 921600 baud to confirm streaming format.
2. **Collect raw data (PC side)**
```bash
cd src
python3 pc_data_logger.py --port /dev/ttyUSB0 --subject 01
# Repeat with `--subject 02`, `--subject 03`, etc., for each participant/session.
```
   - The script writes `subject_01_raw.csv` (and so on) into `./data/`.
3. **Preprocess & feature extraction**
```bash
cd src
python3 -c "from data_loader import load_dataset; from preprocessing import window_and_balance; from feature_extraction import extract_all_features; \
X_raw, y_labels, subject_ids = load_dataset('../data/'); \
X_windows, y_windows, subj_windows = window_and_balance(X_raw, y_labels, subject_ids); \
X_features = extract_all_features(X_windows); \
print('Feature matrix shape:', X_features.shape)"
```
4. **Train & validate**
```bash
cd src
python3 train_pipeline.py --data_folder ../data/ --output_folder ../results/
```
   - This will run 5-Fold CV, LOSO CV, save `best_model.pkl`, and produce:
     - `results/cv_results.csv`
     - `results/loso_results.csv`
   - Metrics are printed to console.
5. **Optional Notebook**
```bash
cd ../notebooks
jupyter notebook train_pipeline.ipynb
```
   - Follow the markdown cells for step-by-step demonstration and visualizations.

## What’s Still Missing / TODO
1. **ESP32 Sketch**
   - ADS1299 advanced calibration (gain settings, input multiplexer) is stubbed.
   - Proper timing synchronization to ensure exactly 8 kHz EMG sampling.
   - Haptic feedback routine (`triggerHaptic(int)`) currently prints a placeholder.
2. **pc_data_logger.py**
   - "Real-time label" input: ability to press a key (e.g., spacebar) to tag "correct" vs. "incorrect" reps during recording.
   - Error handling for serial timeouts or disconnects.
3. **preprocessing.py**
   - Refinement of window-label alignment if label changes mid-window.
   - Better handling for variable-length recordings.
4. **feature_extraction.py**
   - Validation of feature correctness (e.g., verify %MVC computation with actual MVC calibration data).
   - Additional features (e.g., wavelet coefficients) can be added in future.
5. **train_pipeline.py**
   - Hyperparameter search is limited; expand search grid or integrate with Optuna.
   - Support for additional classifiers (e.g., SVM, XGBoost).
   - Logging and checkpointing if training is interrupted.
6. **Notebook**
   - Additional visualizations (ROC curves, t-SNE plots) are not yet implemented.
   - Example dataset is not included; user must populate `./data/` first.

## Contact & License
If you have questions, open an issue or contact the original author.
This code is released under the MIT License (see LICENSE file).
