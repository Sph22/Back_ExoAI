import os
import io
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_PATH = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
METRICS_PATH = os.path.join(DATA_DIR, "training_metrics.json")

# ------------ utilidades de lectura robusta ------------
def _read_csv_bytes_loose(data: bytes) -> pd.DataFrame:
    bio = io.BytesIO(data)
    return pd.read_csv(bio, comment="#", engine="python", on_bad_lines="skip")

def _read_latest_csv(name: str) -> pd.DataFrame:
    """
    Lee data/<name>.csv con estrategias robustas.
    """
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe {path}. Ejecuta /datasets primero.")

    # 1) reproducir lectura por bytes (más tolerante)
    try:
        with open(path, "rb") as f:
            data = f.read()
        df = _read_csv_bytes_loose(data)
        if df is not None and df.shape[1] > 1:
            return df
    except Exception:
        pass

    # 2) autodetección de separador/encoding
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(
                path,
                comment="#",
                engine="python",
                sep=None,
                on_bad_lines="skip",
                encoding=enc,
            )
            if df is not None and df.shape[1] > 1:
                return df
        except Exception:
            continue

    raise ValueError(f"No se pudo leer {path} con ningún método.")

# ------------ mapeo de etiquetas ------------
def map_koi_label(disposition: str) -> str:
    # valores ya vienen como CONFIRMED / CANDIDATE / FALSE POSITIVE
    return str(disposition).upper()

def map_k2_label(disposition: str) -> str:
    d = str(disposition).upper()
    if d in ("REFUTED", "FALSE POSITIVE", "FP"):
        return "FALSE POSITIVE"
    if d == "CONFIRMED":
        return "CONFIRMED"
    if d == "CANDIDATE":
        return "CANDIDATE"
    return d

def map_toi_label(tfopwg_disp: str) -> str:
    d = str(tfopwg_disp).upper()
    if d in ("CP", "KP"):
        return "CONFIRMED"
    if d in ("PC", "APC"):
        return "CANDIDATE"
    if d in ("FP", "FA"):
        return "FALSE POSITIVE"
    return d

# ------------ entrenamiento principal ------------
def train() -> Dict:
    # Cargar datasets más recientes
    koi_df = _read_latest_csv("cumulative")
    toi_df = _read_latest_csv("TOI")
    k2_df  = _read_latest_csv("k2pandc")

    # Filtros / normalización
    if "default_flag" in k2_df.columns:
        k2_df = k2_df[k2_df["default_flag"] == 1].copy()

    # Etiquetas unificadas
    if "koi_disposition" in koi_df.columns:
        koi_df["Label"] = koi_df["koi_disposition"].apply(map_koi_label)
    else:
        raise ValueError("cumulative.csv no contiene 'koi_disposition'.")

    if "disposition" in k2_df.columns:
        k2_df["Label"] = k2_df["disposition"].apply(map_k2_label)
    else:
        raise ValueError("k2pandc.csv no contiene 'disposition'.")

    if "tfopwg_disp" in toi_df.columns:
        toi_df["Label"] = toi_df["tfopwg_disp"].apply(map_toi_label)
    else:
        raise ValueError("TOI.csv no contiene 'tfopwg_disp'.")

    # Selección de columnas (con get para tolerar ausentes)
    koi_selected = pd.DataFrame({
        "period":     koi_df.get("koi_period"),
        "duration":   koi_df.get("koi_duration"),
        "depth":      koi_df.get("koi_depth"),
        "radius":     koi_df.get("koi_prad"),
        "insolation": koi_df.get("koi_insol"),
        "teff":       koi_df.get("koi_steff"),
        "srad":       koi_df.get("koi_srad"),
        "label":      koi_df["Label"],
    })

    k2_selected = pd.DataFrame({
        "period":     k2_df.get("pl_orbper"),
        "duration":   np.nan,
        "depth":      np.nan,
        "radius":     k2_df.get("pl_rade"),
        "insolation": k2_df.get("pl_insol"),
        "teff":       k2_df.get("st_teff"),
        "srad":       k2_df.get("st_rad"),
        "label":      k2_df["Label"],
    })

    toi_selected = pd.DataFrame({
        "period":     toi_df.get("pl_orbper"),
        "duration":   toi_df.get("pl_trandurh"),
        "depth":      toi_df.get("pl_trandep"),
        "radius":     toi_df.get("pl_rade"),
        "insolation": toi_df.get("pl_insol"),
        "teff":       toi_df.get("st_teff"),
        "srad":       toi_df.get("st_rad"),
        "label":      toi_df["Label"],
    })

    data = pd.concat([koi_selected, k2_selected, toi_selected], ignore_index=True)

    # Features / imputación
    features = ["period", "duration", "depth", "radius", "insolation", "teff", "srad"]
    imputer = SimpleImputer(strategy="median")
    data[features] = imputer.fit_transform(data[features])

    # Train / test
    X = data[features].values
    y = data["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    # Guardar bundle
    model_bundle = {"imputer": imputer, "classifier": model, "features": features}
    joblib.dump(model_bundle, MODEL_PATH)

    # Guardar métricas
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "samples_total": int(len(data)),
        "accuracy": acc,
        "classes": sorted(list(set(y))),
        "metrics_path": METRICS_PATH,
    }

if __name__ == "__main__":
    out = train()
    print(out)