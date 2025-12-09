from __future__ import annotations
from data_interface import load_for_model, get_eval_windows
from mnar_blackout_lds import MNARParams, MNARBlackoutLDS
import os
import time
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# 1. Helper: initialize params from data
# ---------------------------------------------------------------------
def init_params_from_data(
    x_t: np.ndarray,
    m_t: np.ndarray,
    latent_dim: int = 8,
    seed: int = 0,
) -> MNARParams:
    """
    Initialize MNARParams using MNARParams.init_random, and optionally
    scale R based on empirical speed variance.
    """
    T, D = x_t.shape

    # Use random initializer
    params = MNARParams.init_random(K=latent_dim, D=D, seed=seed)

    # Compute per-detector variance across observed entries only
    x_obs = np.where(m_t == 0, x_t, np.nan)
    obs_var = np.nanvar(x_obs, axis=0) 

    # If some detectors have no data, fall back to a default variance
    default_var = np.nanmean(obs_var[np.isfinite(obs_var)]) or 4.0
    obs_var = np.where(np.isfinite(obs_var) & (obs_var > 0), obs_var, default_var)

    # Set R to a diagonal with these variances (or scaled)
    params.R = np.diag(obs_var.astype(float))

    return params


# ---------------------------------------------------------------------
# 2. Helper: run EKF + RTS on a possibly masked copy of data
# ---------------------------------------------------------------------

def run_filter_and_smoother(
    model: MNARBlackoutLDS,
    x_t_masked: np.ndarray,
    m_t_masked: np.ndarray,
):
    """
    Run ekf_forward + rts_smoother on the given (possibly masked) sequence.

    Returns
    -------
    ekf_res : dict
    smooth_res : dict
    x_hat : (T, D) array
        Reconstructed speed panel from smoothed latent states.
    """
    ekf_res = model.ekf_forward(x_t=x_t_masked, m_t=m_t_masked)
    smooth_res = model.rts_smoother(ekf_results=ekf_res)
    x_hat = model.reconstruct_from_smoother(mu_smooth=smooth_res["mu_smooth"])
    return ekf_res, smooth_res, x_hat


# ---------------------------------------------------------------------
# 3. Evaluation on blackout windows
# ---------------------------------------------------------------------

def evaluate_on_windows(
    model: MNARBlackoutLDS,
    x_t: np.ndarray,
    m_t: np.ndarray,
    timestamps: np.ndarray,
    detectors: np.ndarray,
    eval_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate imputation + forecasting on the blackout windows in eval_df.

    Assumes eval_df has columns:
      - window_id, detector_id, blackout_start, blackout_end,
        len_steps, test_type, horizon_steps
    """
    # Map timestamps to indices for fast lookup
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    # Map detector_id string to column index
    det_to_idx = {det: i for i, det in enumerate(detectors)}

    # Make sure eval_df has the columns we expect
    required_cols = {
        "window_id",
        "detector_id",
        "blackout_start",
        "blackout_end",
        "len_steps",
        "test_type",
        "horizon_steps",
    }
    missing_cols = required_cols - set(eval_df.columns)
    if missing_cols:
        raise ValueError(f"evaluation_windows missing columns: {missing_cols}")

    results_rows = []

    # Work by window_id so we can run EKF once per blackout scenario
    groups = list(eval_df.groupby("window_id"))
    n_windows = len(groups)

    for idx, (window_id, group) in enumerate(groups, start=1):
        if idx == 1 or idx % 10 == 0 or idx == n_windows:
            print(
                f"[evaluate_on_windows] processing window {idx}/{n_windows} "
                f"(window_id={window_id})"
            )

        # All rows in a group share the same detector + blackout interval
        row0 = group.iloc[0]
        det_id = row0["detector_id"]
        start_ts = row0["blackout_start"]
        end_ts = row0["blackout_end"]

        # Translate to indices
        if det_id not in det_to_idx:
            raise KeyError(f"Detector ID {det_id!r} not found in meta['detectors'].")
        d_idx = det_to_idx[det_id]

        if start_ts not in ts_to_idx or end_ts not in ts_to_idx:
            raise KeyError(
                f"Blackout timestamps {start_ts}–{end_ts} not found in meta['timestamps']."
            )
        t_start = ts_to_idx[start_ts]
        t_end = ts_to_idx[end_ts]

        # Clone full panel and masks
        x_masked = x_t.copy()
        m_masked = m_t.copy()

        # Ground truth values inside blackout
        x_true_blackout = x_t[t_start : t_end + 1, d_idx]

        # Artificially hide the blackout interval for this detector
        x_masked[t_start : t_end + 1, d_idx] = np.nan
        m_masked[t_start : t_end + 1, d_idx] = 1

        # Run EKF + RTS on the masked sequence
        ekf_res, smooth_res, x_hat = run_filter_and_smoother(
            model=model,
            x_t_masked=x_masked,
            m_t_masked=m_masked,
        )

        # -----------------------------------------------------------------
        # Imputation MSE inside blackout
        # -----------------------------------------------------------------
        # Prediction from smoothed states
        x_pred_blackout = x_hat[t_start : t_end + 1, d_idx]

        #  ensure we have finite preds and GT
        mask_finite = np.isfinite(x_true_blackout) & np.isfinite(x_pred_blackout)
        if mask_finite.any():
            mse_impute = float(
                np.mean((x_true_blackout[mask_finite] - x_pred_blackout[mask_finite]) ** 2)
            )
        else:
            mse_impute = np.nan

        results_rows.append(
            {
                "window_id": window_id,
                "detector_id": det_id,
                "metric_type": "impute_mse",
                "horizon_steps": 0,
                "mse": mse_impute,
            }
        )

        # -----------------------------------------------------------------
        # Forecasting MSE at each horizon h
        # -----------------------------------------------------------------
        mu_filt = ekf_res["mu_filt"]
        Sigma_filt = ekf_res["Sigma_filt"]

        # We forecast from the last time index inside the blackout
        start_idx = t_end

        # Loop over forecast rows for this window
        for _, row in group[group["test_type"] == "forecast"].iterrows():
            h = int(row["horizon_steps"])
            mean_x, cov_x = model.k_step_forecast(
                mu_filt=mu_filt,
                Sigma_filt=Sigma_filt,
                start_idx=start_idx,
                k=h,
            )
            # Predicted speed for this detector at time t_end + h
            t_forecast = t_end + h
            if t_forecast >= x_t.shape[0]:
                # Skip if horizon jumps beyond data range
                continue

            x_true_forecast = x_t[t_forecast, d_idx]
            x_pred_forecast = mean_x[d_idx]

            if np.isfinite(x_true_forecast) and np.isfinite(x_pred_forecast):
                mse_forecast = float((x_true_forecast - x_pred_forecast) ** 2)
            else:
                mse_forecast = np.nan

            results_rows.append(
                {
                    "window_id": window_id,
                    "detector_id": det_id,
                    "metric_type": "forecast_mse",
                    "horizon_steps": h,
                    "mse": mse_forecast,
                }
            )

    results_df = pd.DataFrame(results_rows)
    return results_df


# ---------------------------------------------------------------------
# 4. MAR vs MNAR comparison helper
# ---------------------------------------------------------------------

def compare_mar_mnar_results(
    mar_csv: str = "mar_blackout_eval_results.csv",
    mnar_csv: str = "mnar_blackout_eval_results.csv",
) -> None:
    """
    Compare MAR vs MNAR blackout performance.

    Assumes both CSVs have columns:
      - window_id, detector_id, metric_type, horizon_steps, mse
    """
    if not os.path.exists(mar_csv):
        print(f"[compare_mar_mnar_results] MAR file not found: {mar_csv}")
        return
    if not os.path.exists(mnar_csv):
        print(f"[compare_mar_mnar_results] MNAR file not found: {mnar_csv}")
        return

    mar_df = pd.read_csv(mar_csv)
    mnar_df = pd.read_csv(mnar_csv)

    # Rename mse columns so we can distinguish them after the merge
    mar_df = mar_df.rename(columns={"mse": "mse_mar"})
    mnar_df = mnar_df.rename(columns={"mse": "mse_mnar"})

    key_cols = ["window_id", "detector_id", "metric_type", "horizon_steps"]

    merged = pd.merge(
        mar_df,
        mnar_df,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )

    # Avoid divide-by-zero: drop rows where MAR mse <= 0 or NaN
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mse_mar", "mse_mnar"]
    )
    merged = merged[merged["mse_mar"] > 0]

    if merged.empty:
        print("[compare_mar_mnar_results] No valid overlapping rows to compare.")
        return

    # Positive = MNAR better (lower error than MAR)
    merged["improvement_pct"] = (
        100.0 * (merged["mse_mar"] - merged["mse_mnar"]) / merged["mse_mar"]
    )

    print("\n================ MAR vs MNAR comparison ================")
    for (metric, h), group in merged.groupby(["metric_type", "horizon_steps"]):
        imp = group["improvement_pct"]

        frac_better = (imp > 0).mean() * 100.0
        mean_imp = imp.mean()
        median_imp = imp.median()
        p75 = imp.quantile(0.75)
        p90 = imp.quantile(0.90)

        print(f"\nMetric = {metric}, horizon_steps = {h}")
        print(f"  windows: {len(group)}")
        print(f"  mean improvement:   {mean_imp:6.2f} %")
        print(f"  median improvement: {median_imp:6.2f} %")
        print(f"  75th percentile:    {p75:6.2f} %")
        print(f"  90th percentile:    {p90:6.2f} %")
        print(f"  MNAR better on:     {frac_better:6.2f} % of windows")

    print("========================================================\n")


# ---------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------

def main():
    overall_start = time.time()
    # ------------------------------------------------------------
    # Load data using data_interface helper
    # ------------------------------------------------------------
    # x_t: (T, D) speeds with NaNs
    # m_t: (T, D) 1 = missing, 0 = observed
    # meta: timestamps, detectors, dt_minutes
    x_t, m_t, O_t_list, meta = load_for_model()

    timestamps = meta["timestamps"]  
    detectors = meta["detectors"]    

    use_subset = True
    if use_subset:
        steps_per_day = 24 * 12  # assuming 5-min data
        max_days = 24            # Change this to control number of datapoints
        max_T = min(x_t.shape[0], steps_per_day * max_days)
        max_D = min(x_t.shape[1], 60)  # Change the second argument to control number of detectors

        x_t = x_t[:max_T, :max_D].astype(np.float32)
        m_t = m_t[:max_T, :max_D].astype(np.float32)
        timestamps = timestamps[:max_T]
        detectors = detectors[:max_D]

    print("Loaded panel:")
    print(f"  x_t shape: {x_t.shape}")
    print(f"  m_t shape: {m_t.shape}")
    print(f"  #timestamps: {len(timestamps)}, #detectors: {len(detectors)}")

    print("Loaded panel:")
    print(f"  x_t shape: {x_t.shape}")
    print(f"  m_t shape: {m_t.shape}")
    print(f"  #timestamps: {len(timestamps)}, #detectors: {len(detectors)}")

    # ------------------------------------------------------------
    # Shared settings (heavier data → slightly smaller K, fewer EM iters)
    # ------------------------------------------------------------
    latent_dim = 15          # latent dim (smaller to keep runtime sane)
    num_em_iters_mar = 7     # MAR EM iterations
    num_em_iters_mnar = 7    # MNAR EM iterations
    max_windows = 10         # number of blackout windows for eval when enabled
    
    # ------------------------------------------------------------
    # Load evaluation windows manifest and subset to match panel
    # ------------------------------------------------------------
    eval_df = get_eval_windows(as_dataframe=True)
    print("\nLoaded evaluation windows:")
    print(eval_df.head())

    ts_set = set(timestamps)
    det_set = set(detectors)

    eval_df = eval_df[
        eval_df["detector_id"].isin(det_set)
        & eval_df["blackout_start"].isin(ts_set)
        & eval_df["blackout_end"].isin(ts_set)
    ].copy()

    print("\nEval windows after filtering to subset:")
    print(eval_df.head())
    print(f"#eval windows: {len(eval_df)}")

    unique_ids = sorted(eval_df["window_id"].unique())
    keep_ids = unique_ids[:max_windows]
    eval_subset = eval_df[eval_df["window_id"].isin(keep_ids)].copy()

    print(
        f"\nUsing {len(keep_ids)} window_ids "
        f"({len(eval_subset)} rows out of {len(eval_df)} total eval rows)."
    )


    # ------------------------------------------------------------
    # MAR-style run: EM without updating phi
    # ------------------------------------------------------------
    print("\n========== MAR-style LDS run (update_phi=False) ==========")
    params_mar = init_params_from_data(x_t=x_t, m_t=m_t, latent_dim=latent_dim, seed=0)
    model_mar = MNARBlackoutLDS(params=params_mar)

    print(f"Training MAR model with K={latent_dim}, EM iters={num_em_iters_mar}")
    start_mar = time.time()
    _ = model_mar.em_train(
        x_t=x_t,
        m_t=m_t,
        num_iters=num_em_iters_mar,
        update_phi=False,   # <- key: ignore missingness mechanism (MAR)
        phi_steps=0,
        phi_lr=0.0,
        verbose=True,
        convergence_tol=1e-2,
    )

    mar_time = time.time() - start_mar
    print(f"MAR run finished in {int(mar_time // 60)} min {int(mar_time % 60)} sec")

    # Always run blackout evaluation and save CSV for later analysis
    results_mar = evaluate_on_windows(
        model=model_mar,
        x_t=x_t,
        m_t=m_t,
        timestamps=timestamps,
        detectors=detectors,
        eval_df=eval_subset,
    )

    mar_csv = "mar_blackout_eval_results.csv"
    results_mar.to_csv(mar_csv, index=False)
    print(f"\n[MAR] Saved per-window MAR results to {mar_csv}")


    print("\nMAR overall metric summary:")
    print(results_mar.groupby(["metric_type", "horizon_steps"])["mse"].mean())

    # ------------------------------------------------------------
    # MNAR run: EM with logistic missingness updates
    # ------------------------------------------------------------
    print("\n========== MNAR LDS run (update_phi=True) ==========")
    params_mnar = init_params_from_data(x_t=x_t, m_t=m_t, latent_dim=latent_dim, seed=0)
    model_mnar = MNARBlackoutLDS(params=params_mnar)

    print(f"Training MNAR model with K={latent_dim}, EM iters={num_em_iters_mnar}")
    start_mnar = time.time()
    _ = model_mnar.em_train(
        x_t=x_t,
        m_t=m_t,
        num_iters=num_em_iters_mnar,
        update_phi=True,    # <- key: learn missingness mechanism (MNAR)
        phi_steps=3,
        phi_lr=3e-3,
        verbose=True,
        convergence_tol=1e-3,
    )

    mnar_time = time.time() - start_mnar
    print(f"MNAR run finished in {int(mnar_time // 60)} min {int(mnar_time % 60)} sec")

    # Always run blackout evaluation and save CSV for later analysis
    results_mnar = evaluate_on_windows(
        model=model_mnar,
        x_t=x_t,
        m_t=m_t,
        timestamps=timestamps,
        detectors=detectors,
        eval_df=eval_subset,
    )

    mnar_csv = "mnar_blackout_eval_results.csv"
    results_mnar.to_csv(mnar_csv, index=False)
    print(f"\n[MNAR] Saved per-window MNAR results to {mnar_csv}")

    print("\nMNAR overall metric summary:")
    print(results_mnar.groupby(["metric_type", "horizon_steps"])["mse"].mean())

    # --------------------------------------------------------
    # Comparison: MAR vs MNAR
    # --------------------------------------------------------
    compare_mar_mnar_results(mar_csv=mar_csv, mnar_csv=mnar_csv)

    # ------------------------------------------------------------
    # Overall runtime
    # ------------------------------------------------------------
    total_time = time.time() - overall_start
    tot_mins, tot_secs = divmod(int(total_time), 60)
    print(f"\nTotal script runtime: {tot_mins} min {tot_secs} sec")


if __name__ == "__main__":
    main()