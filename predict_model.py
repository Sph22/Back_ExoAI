import os
from typing import Dict, List, Union

import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "exoplanet_model.pkl")

_model_bundle = None

def _load_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No existe el modelo en {MODEL_PATH}. Entrena primero.")
        _model_bundle = joblib.load(MODEL_PATH)
    return _model_bundle

def _ensure_order(x: Dict[str, float], features: List[str]) -> List[float]:
    return [float(x.get(f, np.nan)) for f in features]

def predict_one(payload: Dict[str, float]) -> Union[int, float, str]:
    bundle = _load_bundle()
    imputer = bundle["imputer"]
    clf = bundle["classifier"]
    features = bundle.get("features", ["period","duration","depth","radius","insolation","teff","srad"])

    X = np.array([_ensure_order(payload, features)], dtype=float)
    X = imputer.transform(X)
    y_hat = clf.predict(X)
    return y_hat.item() if hasattr(y_hat, "item") else y_hat[0]

def predict_batch(payload: List[Dict[str, float]]) -> List[Union[int, float, str]]:
    bundle = _load_bundle()
    imputer = bundle["imputer"]
    clf = bundle["classifier"]
    features = bundle.get("features", ["period","duration","depth","radius","insolation","teff","srad"])

    X = np.array([_ensure_order(row, features) for row in payload], dtype=float)
    X = imputer.transform(X)
    preds = clf.predict(X)
    return [p.item() if hasattr(p, "item") else p for p in preds]