import os
from functools import lru_cache
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd

# Columnas esperadas por el pipeline (en este orden)
FEATURES: List[str] = ["period", "duration", "depth", "radius", "insolation", "teff", "srad"]

MODEL_PATH = os.getenv("MODEL_PATH", "exoplanet_model.pkl")

@lru_cache(maxsize=1)
def _load_bundle():
    """Carga y cachea el bundle (imputer + classifier)."""
    bundle = joblib.load(MODEL_PATH)
    imputer = bundle["imputer"]
    model = bundle["classifier"]
    return imputer, model

def _to_dataframe(payload: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
    """Normaliza la entrada a DataFrame con las columnas en FEATURES."""
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, pd.DataFrame):
        df = payload.copy()
    else:
        raise TypeError("payload debe ser dict o pandas.DataFrame")
    # Reordenar/forzar columnas
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        # Si faltan columnas, créalas como NaN para que el imputador las resuelva
        for c in missing:
            df[c] = np.nan
    return df[FEATURES]

def predict_one(payload: Dict) -> Union[int, float, str]:
    """Predicción para un solo registro (dict con las FEATURES)."""
    imputer, model = _load_bundle()
    X = _to_dataframe(payload)
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)
    y_hat = model.predict(X_imp.values)[0]
    return y_hat.item() if hasattr(y_hat, "item") else y_hat

def predict_batch(payload: Union[List[Dict], pd.DataFrame]) -> List[Union[int, float, str]]:
    """Predicción para lote: lista de dicts o DataFrame."""
    imputer, model = _load_bundle()
    if isinstance(payload, list):
        df = pd.DataFrame(payload)
    else:
        df = payload.copy()
    X = _to_dataframe(df)
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)
    y_hat = model.predict(X_imp.values)
    return [v.item() if hasattr(v, "item") else v for v in y_hat]

def predict_from_csv(csv_path: str) -> List[Union[int, float, str]]:
    """Helper para predecir desde un CSV. No se ejecuta en import."""
    df = pd.read_csv(csv_path)
    return predict_batch(df)