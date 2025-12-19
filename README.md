# Modelling Information Blackouts in MNAR Time Series (Traffic Sensors)

This repo studies traffic sensor blackouts (contiguous missing intervals) and compares MAR vs MNAR state-space models for:
- Blackout imputation (reconstruct values inside blackout windows)
- Post-blackout forecasting (predict 1 / 3 / 6 steps after a blackout ends)

Core idea: treat the missingness mask as an informative observation channel (MNAR), not just something to ignore or impute away.

## How to run

### 1) Install dependencies
pip install -r requirements.txt

### 2) Put the data in place
Ensure data/ contains the cleaned Seattle Loop-wide panel used by data_interface.py (parquet).

### 3) Build artifacts (Seattle Loop)
Run notebooks in this order:
1. 01_load_and_clean.ipynb
2. 02_missingness_eda.ipynb
3. 03_blackout_detection.ipynb
4. 04_build_xt_mt.ipynb
5. 05_phi_features.ipynb (optional)
6. 06_evaluation_windows.ipynb

### 4) Train + evaluate models
jupyter notebook
Open and run:
- main.ipynb
