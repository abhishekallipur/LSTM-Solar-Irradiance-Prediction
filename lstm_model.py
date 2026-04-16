"""
Peak-aware CNN-BiLSTM-Attention model for next-hour GHI forecasting (TensorFlow/Keras).

Why this version improves over a plain LSTM:
1) Better features: time cycles, lags, rolling stats, and ramps improve cloud-change awareness.
2) Better scaling: train-only fitting + robust input scaling prevents outlier distortion.
3) Better objective: weighted Huber emphasizes peaks/ramps and reduces oversmoothing bias.
4) Better architecture: Conv1D + BiLSTM + attention captures local spikes + longer context.
5) Better physics handling: night-time gating avoids false positive irradiance at night.

This script is end-to-end runnable on a normal laptop.
"""

import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------
# NSRDB loading
# ---------------------------
def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _find_column(df: pd.DataFrame, aliases: List[str]) -> str:
    lookup = {_normalize_name(c): c for c in df.columns}
    for alias in aliases:
        key = _normalize_name(alias)
        if key in lookup:
            return lookup[key]
    return ""


def _load_single_nsrdb_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    metadata_df = pd.read_csv(csv_path, nrows=1)
    metadata = {str(k): str(v) for k, v in metadata_df.iloc[0].to_dict().items()}
    data_df = pd.read_csv(csv_path, skiprows=2)
    return data_df, metadata


def _load_csv_or_folder(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in folder: {path}")
    else:
        files = [path]

    frames = []
    metadata: Dict[str, str] = {}
    for i, file_path in enumerate(files):
        frame, meta = _load_single_nsrdb_csv(file_path)
        frames.append(frame)
        if i == 0:
            metadata = meta

    df = pd.concat(frames, ignore_index=True)
    return df, metadata


def _calculate_solar_zenith_angle(
    timestamps: pd.Series, latitude: float, longitude: float, timezone_offset_hours: float
) -> np.ndarray:
    day_of_year = timestamps.dt.dayofyear.to_numpy(dtype=np.float64)
    hour = timestamps.dt.hour.to_numpy(dtype=np.float64)
    minute = timestamps.dt.minute.to_numpy(dtype=np.float64)

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    time_offset = eqtime + 4.0 * longitude - 60.0 * timezone_offset_hours
    true_solar_time = (hour * 60.0 + minute + time_offset) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0
    hour_angle[hour_angle < -180.0] += 360.0

    lat_rad = np.deg2rad(latitude)
    ha_rad = np.deg2rad(hour_angle)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(ha_rad)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_zenith))


# ---------------------------
# Feature engineering
# ---------------------------
def build_feature_table(path: str) -> pd.DataFrame:
    raw_df, metadata = _load_csv_or_folder(path)

    aliases = {
        "ghi": ["GHI", "Global Horizontal Irradiance"],
        "temp": ["Temperature", "Air Temperature", "Temp"],
        "rh": ["Relative Humidity", "RH", "Humidity"],
        "wind": ["Wind Speed", "WindSpeed"],
        "pressure": ["Pressure", "Surface Pressure", "Atmospheric Pressure"],
        "zenith": ["Solar Zenith Angle", "Zenith Angle", "Solar Zenith"],
    }

    selected = {}
    for name, name_aliases in aliases.items():
        col = _find_column(raw_df, name_aliases)
        if not col and name != "zenith":
            raise ValueError(f"Missing required column for '{name}'. Available: {list(raw_df.columns)}")
        selected[name] = col

    required_time = ["Year", "Month", "Day", "Hour", "Minute"]
    if not all(c in raw_df.columns for c in required_time):
        raise ValueError("NSRDB file must include Year, Month, Day, Hour, Minute columns.")

    dt = pd.to_datetime(
        {
            "year": raw_df["Year"],
            "month": raw_df["Month"],
            "day": raw_df["Day"],
            "hour": raw_df["Hour"],
            "minute": raw_df["Minute"],
        },
        errors="coerce",
    )
    if dt.isna().any():
        raise ValueError("Could not parse timestamps from Year/Month/Day/Hour/Minute.")

    df = pd.DataFrame(index=np.arange(len(raw_df)))
    df["timestamp"] = dt
    df["ghi"] = pd.to_numeric(raw_df[selected["ghi"]], errors="coerce")
    df["temp"] = pd.to_numeric(raw_df[selected["temp"]], errors="coerce")
    df["rh"] = pd.to_numeric(raw_df[selected["rh"]], errors="coerce")
    df["wind"] = pd.to_numeric(raw_df[selected["wind"]], errors="coerce")
    df["pressure"] = pd.to_numeric(raw_df[selected["pressure"]], errors="coerce")

    if selected["zenith"]:
        df["zenith"] = pd.to_numeric(raw_df[selected["zenith"]], errors="coerce")
    else:
        lat = float(metadata["Latitude"])
        lon = float(metadata["Longitude"])
        tz = float(metadata["Local Time Zone"])
        df["zenith"] = _calculate_solar_zenith_angle(df["timestamp"], lat, lon, tz)

    # Time cyclical features
    hour = df["timestamp"].dt.hour.astype(float)
    dayofyear = df["timestamp"].dt.dayofyear.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.0)

    # Physics-aware/night features
    cosz = np.cos(np.deg2rad(df["zenith"].clip(lower=0, upper=180)))
    df["cos_zenith"] = np.clip(cosz, 0.0, 1.0)
    df["is_daylight"] = (df["zenith"] < 90.0).astype(float)

    # Lags and ramps for cloud-driven changes
    df["ghi_lag_1"] = df["ghi"].shift(1)
    df["ghi_lag_2"] = df["ghi"].shift(2)
    df["ghi_lag_24"] = df["ghi"].shift(24)
    df["ghi_diff_1"] = df["ghi"] - df["ghi"].shift(1)
    df["ghi_diff_3"] = df["ghi"] - df["ghi"].shift(3)
    df["temp_diff_1"] = df["temp"] - df["temp"].shift(1)
    df["rh_diff_1"] = df["rh"] - df["rh"].shift(1)

    # Rolling context
    df["ghi_roll_mean_3"] = df["ghi"].rolling(3).mean()
    df["ghi_roll_mean_6"] = df["ghi"].rolling(6).mean()
    df["ghi_roll_mean_24"] = df["ghi"].rolling(24).mean()
    df["ghi_roll_std_6"] = df["ghi"].rolling(6).std()
    df["ghi_roll_std_24"] = df["ghi"].rolling(24).std()
    df["wind_roll_mean_6"] = df["wind"].rolling(6).mean()
    df["rh_roll_mean_6"] = df["rh"].rolling(6).mean()

    # Fill missing values from lag/rolling operations
    df = df.ffill().bfill()
    return df


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    w_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    daylight_test_next: np.ndarray
    target_scaler: StandardScaler
    feature_names: List[str]
    train_peak_threshold: float


def make_supervised_dataset(
    df: pd.DataFrame,
    sequence_length: int = 48,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> DatasetBundle:
    # Features used by model
    feature_cols = [
        "ghi", "temp", "rh", "wind", "pressure", "zenith", "cos_zenith", "is_daylight",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos",
        "ghi_lag_1", "ghi_lag_2", "ghi_lag_24",
        "ghi_diff_1", "ghi_diff_3", "temp_diff_1", "rh_diff_1",
        "ghi_roll_mean_3", "ghi_roll_mean_6", "ghi_roll_mean_24",
        "ghi_roll_std_6", "ghi_roll_std_24", "wind_roll_mean_6", "rh_roll_mean_6",
    ]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Feature '{col}' missing in engineered dataframe.")

    values = df[feature_cols].to_numpy(dtype=np.float32)
    target_raw = df["ghi"].to_numpy(dtype=np.float32)
    daylight = df["is_daylight"].to_numpy(dtype=np.float32)
    ghi_diff_abs = np.abs(df["ghi_diff_1"].to_numpy(dtype=np.float32))

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= sequence_length + 2:
        raise ValueError("Dataset split too small for chosen sequence length.")

    # Fit scalers on train only (prevents leakage)
    X_scaler = RobustScaler()
    X_scaler.fit(values[:n_train])
    X_scaled = X_scaler.transform(values).astype(np.float32)

    # Target scaler fit on daytime training targets (better dynamic range for useful signal)
    train_day_mask = (daylight[:n_train] > 0.5)
    y_scaler = StandardScaler()
    y_scaler.fit(target_raw[:n_train][train_day_mask].reshape(-1, 1))
    y_scaled = y_scaler.transform(target_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)

    # Build sequences and aligned metadata
    X_seq, y_seq = [], []
    y_raw_seq, daylight_next, ramp_next, last_ghi = [], [], [], []

    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i - sequence_length:i, :])
        y_seq.append(y_scaled[i])
        y_raw_seq.append(target_raw[i])
        daylight_next.append(daylight[i])
        ramp_next.append(ghi_diff_abs[i])
        last_ghi.append(target_raw[i - 1])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32).reshape(-1, 1)
    y_raw_seq = np.asarray(y_raw_seq, dtype=np.float32).reshape(-1, 1)
    daylight_next = np.asarray(daylight_next, dtype=np.float32).reshape(-1, 1)
    ramp_next = np.asarray(ramp_next, dtype=np.float32).reshape(-1, 1)
    last_ghi = np.asarray(last_ghi, dtype=np.float32).reshape(-1, 1)

    # Convert original split points to sequence indices
    train_end = n_train - sequence_length
    val_end = n_train + n_val - sequence_length

    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    y_test_raw = y_raw_seq[val_end:]
    daylight_test_next = daylight_next[val_end:]

    y_train_raw = y_raw_seq[:train_end]
    ramp_train = ramp_next[:train_end]
    last_ghi_train = last_ghi[:train_end]

    # Peak/ramp-aware sample weights
    peak_thr = float(np.percentile(y_train_raw[y_train_raw > 0], 90))
    ramp_thr = float(np.percentile(ramp_train, 85))

    w_train = np.ones_like(y_train_raw, dtype=np.float32)
    w_train += 1.6 * (y_train_raw >= peak_thr).astype(np.float32)  # emphasize peaks
    w_train += 1.2 * (ramp_train >= ramp_thr).astype(np.float32)   # emphasize sudden changes
    w_train += 0.6 * (last_ghi_train >= peak_thr).astype(np.float32)
    w_train *= np.where(y_train_raw < 5.0, 0.25, 1.0).astype(np.float32)  # downweight easy nighttime zeros
    w_train = np.clip(w_train, 0.2, 4.5).reshape(-1)

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        y_test_raw=y_test_raw,
        daylight_test_next=daylight_test_next.reshape(-1),
        target_scaler=y_scaler,
        feature_names=feature_cols,
        train_peak_threshold=peak_thr,
    )


# ---------------------------
# Model
# ---------------------------
def build_cnn_bilstm_attention_model(
    sequence_length: int,
    n_features: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    inp = keras.Input(shape=(sequence_length, n_features), name="history")
    x = layers.LayerNormalization()(inp)

    # Local temporal patterns (spikes/ramp segments)
    x = layers.Conv1D(64, kernel_size=3, padding="causal", activation="swish")(x)
    x = layers.Conv1D(64, kernel_size=5, padding="causal", dilation_rate=2, activation="swish")(x)
    x = layers.Dropout(0.15)(x)

    # Longer dependencies
    x = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True, dropout=0.20, recurrent_dropout=0.0)
    )(x)

    # Self-attention to focus on important timesteps
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=24, dropout=0.10)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Robust temporal summary
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    x = layers.Dense(128, activation="swish")(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="swish")(x)
    out = layers.Dense(1, name="ghi_next")(x)

    model = keras.Model(inputs=inp, outputs=out, name="cnn_bilstm_attention_ghi")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(delta=1.0),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


# ---------------------------
# Train / evaluate / plot
# ---------------------------
def inverse_target(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_threshold: float) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    peak_mask = y_true >= peak_threshold
    if np.any(peak_mask):
        peak_mae = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask]))
    else:
        peak_mae = np.nan

    return {"rmse": rmse, "mae": mae, "r2": r2, "peak_mae": peak_mae}


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float], n_hours: int = 400) -> None:
    n = min(n_hours, len(y_true))
    xs = np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)

    axes[0].plot(xs, y_true[:n], label="Actual GHI", color="#1f77b4", linewidth=1.9)
    axes[0].plot(xs, y_pred[:n], label="Predicted GHI", color="#ff7f0e", linewidth=1.9, alpha=0.95)
    axes[0].fill_between(xs, y_true[:n], y_pred[:n], color="#ff7f0e", alpha=0.12, label="Error")
    axes[0].set_title("Next-hour GHI Forecast (first test segment)")
    axes[0].set_ylabel("GHI (W/m^2)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper right")

    # Scatter with ideal line to inspect peak underestimation bias
    axes[1].scatter(y_true, y_pred, s=8, alpha=0.25, color="#2ca02c")
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[1].plot([0, max_v], [0, max_v], "r--", linewidth=1.2, label="Ideal y=x")
    axes[1].set_title("Prediction calibration (all test points)")
    axes[1].set_xlabel("Actual GHI (W/m^2)")
    axes[1].set_ylabel("Predicted GHI (W/m^2)")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(loc="upper left")

    txt = (
        f"RMSE: {metrics['rmse']:.2f} W/m^2\n"
        f"MAE: {metrics['mae']:.2f} W/m^2\n"
        f"R^2: {metrics['r2']:.3f}\n"
        f"Peak-MAE: {metrics['peak_mae']:.2f} W/m^2"
    )
    fig.text(
        0.79,
        0.83,
        txt,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.88, "edgecolor": "#666"},
    )
    fig.tight_layout()
    plt.show()


def train_and_evaluate(
    data_path: str,
    sequence_length: int = 48,
    epochs: int = 60,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    no_plot: bool = False,
) -> None:
    set_seed(seed)
    tf.keras.backend.clear_session()

    df = build_feature_table(data_path)
    bundle = make_supervised_dataset(df, sequence_length=sequence_length)

    model = build_cnn_bilstm_attention_model(
        sequence_length=sequence_length,
        n_features=bundle.X_train.shape[-1],
        learning_rate=learning_rate,
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        bundle.X_train,
        bundle.y_train,
        sample_weight=bundle.w_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
    )

    y_pred_scaled = model.predict(bundle.X_test, batch_size=batch_size, verbose=0).reshape(-1)
    y_true_scaled = bundle.y_test.reshape(-1)

    y_pred = inverse_target(y_pred_scaled, bundle.target_scaler)
    y_true = inverse_target(y_true_scaled, bundle.target_scaler)

    # Physics-informed postprocessing: no irradiance at night
    y_pred = np.clip(y_pred, 0.0, None)
    y_pred[bundle.daylight_test_next < 0.5] = 0.0

    metrics = evaluate_metrics(y_true, y_pred, peak_threshold=bundle.train_peak_threshold)
    print("\nTest Metrics:")
    print(f"RMSE: {metrics['rmse']:.2f} W/m^2")
    print(f"MAE : {metrics['mae']:.2f} W/m^2")
    print(f"R^2 : {metrics['r2']:.4f}")
    print(f"Peak-MAE (>=P90 train daytime): {metrics['peak_mae']:.2f} W/m^2")

    # Daylight-only diagnostic
    day_mask = bundle.daylight_test_next >= 0.5
    if np.any(day_mask):
        day_rmse = np.sqrt(mean_squared_error(y_true[day_mask], y_pred[day_mask]))
        day_mae = mean_absolute_error(y_true[day_mask], y_pred[day_mask])
        print(f"Daylight RMSE: {day_rmse:.2f} W/m^2")
        print(f"Daylight MAE : {day_mae:.2f} W/m^2")

    if not no_plot:
        plot_predictions(y_true, y_pred, metrics=metrics, n_hours=400)

    # Optional: visualize training curves
    if not no_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"], label="Train loss")
        plt.plot(history.history["val_loss"], label="Val loss")
        plt.title("Training/Validation Loss (Huber)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peak-aware TensorFlow model for NSRDB hourly GHI prediction."
    )
    parser.add_argument("--data-path", type=str, default="dataset", help="CSV file or folder of NSRDB CSVs.")
    parser.add_argument("--sequence-length", type=int, default=48, help="History window in hours.")
    parser.add_argument("--epochs", type=int, default=60, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()

