# Modelling Information Blackouts in MNAR Time Series (Traffic Sensors)

This repo studies **traffic sensor blackouts** (contiguous missing intervals) and compares **MAR vs MNAR** state-space models for:

- **Blackout imputation**: reconstruct values *inside* blackout windows  
- **Post‑blackout forecasting**: predict **+1 / +3 / +6** steps after a blackout ends

**Core idea:** treat the missingness mask as an *informative observation channel (MNAR)*, not something to ignore or impute away.

---

## What’s inside

### Models
- **LOCF baseline** (last observation carried forward)
- **MAR LDS / Kalman**: linear Gaussian state-space model with **masked observations** (missing entries are skipped)
- **MNAR LDS (Blackouts-as-signal)**: same LDS + logistic missingness model  
  \[ p(m_{t,d}=1 \mid z_t)=\sigma(\phi_d^\top z_t) \]  
  Inference uses **EKF + RTS** and training uses **EM**.

### Evaluation tasks
- **Imputation inside blackouts**: MAE / RMSE (optionally CRPS)
- **Forecast after blackout**: MAE / RMSE at **k ∈ {1,3,6}** (optionally CRPS)
- **Ablation**: MNAR model with missingness block removed (e.g., **Φ fixed to 0**) to quantify the value of “blackouts as signal.”

---

## Quick start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Put the data in place
This repo expects a cleaned **Seattle Loop 2015** panel (5‑min speeds) written as **parquet**.

Place it under:
```
data/
  seattle_loop_panel.parquet   # (or whatever your data_interface.py expects)
```

> If you change filenames/paths, update `data_interface.py` accordingly.

### 3) Build artifacts (Seattle Loop)
Run notebooks in this order:

1. `01_load_and_clean.ipynb`
2. `02_missingness_eda.ipynb`
3. `03_blackout_detection.ipynb`
4. `04_build_xt_mt.ipynb`
5. `05_phi_features.ipynb` *(optional)*
6. `06_evaluation_windows.ipynb`

These notebooks produce:
- `x_t`: speed panel (T×D)
- `m_t`: missingness masks (T×D, 1 = missing)
- blackout windows and evaluation splits

### 4) Train + evaluate models
```bash
jupyter notebook
```
Open and run:
- `main.ipynb`

---

## Repository structure (typical)

- `data_interface.py` — load panel data + common preprocessing utilities  
- `blackout_detection.py` — blackout window detection logic  
- `evaluation.py` — metric computation (impute + forecast)  
- `mnar_blackout_lds.py` — MNAR LDS (EKF + RTS + EM)  
- `mar_lds.py` — MAR LDS / Kalman + EM baseline  
- `notebooks/` — the pipeline notebooks listed above

(Names may vary depending on your current repo state.)

---

## Notes on the MNAR model

We augment the standard LDS:
- Dynamics: \( z_t \sim \mathcal{N}(A z_{t-1}, Q) \)
- Emissions: \( x_t \sim \mathcal{N}(C z_t, R) \)

with a **state‑dependent missingness mechanism**:
- Missingness: \( p(m_{t,d}=1 \mid z_t)=\sigma(\phi_d^\top z_t) \)

During filtering, the model performs an EKF-style update using:
1) observed speed entries (standard LDS update), and  
2) the missingness mask as a pseudo‑observation (MNAR signal).

---

## Metrics

### Imputation inside blackout windows
- **MAE**, **RMSE** against held‑out ground truth values inside blackout windows.

### Post‑blackout forecasting
- For each blackout end time \(b\), evaluate forecasts at horizons \(k \in \{1,3,6\}\):
  - \(\hat{x}_{b+k} = C\,\mu_{b+k\mid b}\)
- Report MAE/RMSE (optionally CRPS).

---

## Extensions (optional)
- Inject synthetic blackouts on **METR‑LA / PEMS‑BAY** to test generality.
- Add simple calendar features \(\phi_t\) (hour, day‑of‑week, weekend) to the state dynamics or as side inputs.

---

## Team
- **Allan Ma** — literature review, model building  
- **Aman Sunesh** — EDA, evaluation  
- **Siddarth Nilol** — data preprocessing, report writing

---

## Project context (DS‑GA 1018)
This work was developed as part of **DS‑GA 1018: Probabilistic Time Series Analysis** (NYU), focusing on MNAR modelling for structured sensor outages.
