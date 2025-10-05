import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ===== Config =====
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_PATH = os.getenv("MODEL_PATH", "exoplanet_model.pkl")

FEATURES: List[str] = [
    "period", "duration", "depth", "radius", "insolation", "teff", "srad"
]

def _latest_path(prefix: str) -> str:
    """Devuelve data/<prefix>.csv (archivo 'latest' escrito por fetch_datasets)."""
    path = os.path.join(DATA_DIR, f"{prefix}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta primero la actualización de datasets.")
    return path

def _map_labels(koi_df: pd.DataFrame, k2_df: pd.DataFrame, toi_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Kepler KOI: koi_disposition -> ya en CONFIRMED / CANDIDATE / FALSE POSITIVE
    koi_df = koi_df.copy()
    if "koi_disposition" not in koi_df.columns:
        raise KeyError("koi_disposition no existe en cumulative.csv")
    koi_df["Label"] = koi_df["koi_disposition"].astype(str).str.upper()

    # K2: disposition puede incluir REFUTED
    k2_df = k2_df.copy()
    if "disposition" not in k2_df.columns:
        raise KeyError("disposition no existe en k2pandc.csv")
    def map_k2(d: str) -> str:
        d = str(d).upper()
        if d in ("REFUTED", "FALSE POSITIVE"):
            return "FALSE POSITIVE"
        if d == "CONFIRMED":
            return "CONFIRMED"
        if d == "CANDIDATE":
            return "CANDIDATE"
        return d
    k2_df["Label"] = k2_df["disposition"].apply(map_k2)

    # TESS: tfopwg_disp -> CP,KP (confirmed); PC,APC (candidate); FP,FA (false pos)
    toi_df = toi_df.copy()
    if "tfopwg_disp" not in toi_df.columns:
        raise KeyError("tfopwg_disp no existe en TOI.csv")
    def map_toi(d: str) -> str:
        d = str(d).upper()
        if d in ("CP", "KP"):
            return "CONFIRMED"
        if d in ("PC", "APC"):
            return "CANDIDATE"
        if d in ("FP", "FA"):
            return "FALSE POSITIVE"
        return d
    toi_df["Label"] = toi_df["tfopwg_disp"].apply(map_toi)

    return koi_df, k2_df, toi_df

def _select_columns(koi_df: pd.DataFrame, k2_df: pd.DataFrame, toi_df: pd.DataFrame) -> pd.DataFrame:
    # KOI (Kepler)
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

    # K2 (faltan duración y profundidad, las dejamos en NaN)
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

    # TOI (TESS)
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
    return data

def train() -> Dict:
    """
    Entrena el modelo leyendo:
      - data/cumulative.csv
      - data/k2pandc.csv
      - data/TOI.csv
    Guarda el bundle en MODEL_PATH.
    Devuelve métricas y rutas usadas.
    """
    # 1) Cargar datasets “latest”
    koi_path = _latest_path("cumulative")
    k2_path  = _latest_path("k2pandc")
    toi_path = _latest_path("TOI")

    print("Loading datasets...")
    koi_df = pd.read_csv(koi_path, comment="#")
    k2_df  = pd.read_csv(k2_path,  comment="#")
    toi_df = pd.read_csv(toi_path, comment="#")
    print(f"Kepler KOI entries: {len(koi_df)}, K2 entries: {len(k2_df)}, TESS entries: {len(toi_df)}")

    # 2) Filtro K2 default_flag == 1 (si existe)
    if "default_flag" in k2_df.columns:
        k2_df = k2_df[k2_df["default_flag"] == 1].copy()
        print(f"K2 entries after default_flag filter: {len(k2_df)}")

    # 3) Mapear etiquetas a un esquema común
    koi_df, k2_df, toi_df = _map_labels(koi_df, k2_df, toi_df)

    # 4) Selección y renombrado de columnas
    data = _select_columns(koi_df, k2_df, toi_df)
    print(f"Total combined entries: {len(data)}")

    # 5) Imputación
    features = FEATURES
    imputer = SimpleImputer(strategy="median")
    data[features] = imputer.fit_transform(data[features])

    # 6) Split
    X = data[features].values
    y = data["label"].astype(str).values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 7) Modelo
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # 8) Métricas
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    print(f"Accuracy on test data: {accuracy:.4f}")
    print("Classification report:\n", report)

    # 9) Guardar bundle
    model_bundle = {"imputer": imputer, "classifier": model}
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

    return {
        "data_paths": {"koi": koi_path, "k2": k2_path, "toi": toi_path},
        "samples": {"total": int(len(data)), "train": int(len(X_train)), "test": int(len(X_test))},
        "accuracy": float(accuracy),
        "model_path": MODEL_PATH,
    }

if __name__ == "__main__":
    # Entrenamiento manual local
    out = train()
    print(out)