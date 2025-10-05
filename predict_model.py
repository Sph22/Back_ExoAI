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

def predict_one(payload: Dict[str, float]) -> Dict[str, Union[str, float, Dict]]:
    """
    Realiza una predicción y retorna la clase, confianza y probabilidades
    
    Returns:
        {
            "prediction": "CANDIDATE",
            "confidence_percentage": 87.45,
            "probabilities": {
                "CANDIDATE": 87.45,
                "CONFIRMED": 8.32,
                "FALSE POSITIVE": 4.23
            }
        }
    """
    bundle = _load_bundle()
    imputer = bundle["imputer"]
    clf = bundle["classifier"]
    features = bundle.get("features", ["period","duration","depth","radius","insolation","teff","srad"])

    X = np.array([_ensure_order(payload, features)], dtype=float)
    X = imputer.transform(X)
    
    # Predicción
    y_hat = clf.predict(X)[0]
    
    # Probabilidades
    y_proba = clf.predict_proba(X)[0]
    
    # Encontrar índice de la clase predicha
    class_idx = list(clf.classes_).index(y_hat)
    confidence = float(y_proba[class_idx]) * 100
    
    # Construir diccionario de probabilidades
    probabilities = {
        str(cls): round(float(prob) * 100, 2) 
        for cls, prob in zip(clf.classes_, y_proba)
    }
    
    return {
        "prediction": str(y_hat),
        "confidence_percentage": round(confidence, 2),
        "probabilities": probabilities
    }

def predict_batch(payload: List[Dict[str, float]]) -> List[Dict[str, Union[str, float, Dict]]]:
    """
    Realiza predicciones en batch
    
    Returns:
        Lista de diccionarios con predicciones y probabilidades
    """
    bundle = _load_bundle()
    imputer = bundle["imputer"]
    clf = bundle["classifier"]
    features = bundle.get("features", ["period","duration","depth","radius","insolation","teff","srad"])

    X = np.array([_ensure_order(row, features) for row in payload], dtype=float)
    X = imputer.transform(X)
    
    # Predicciones y probabilidades
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)
    
    results = []
    for y_hat, y_proba in zip(predictions, probabilities):
        class_idx = list(clf.classes_).index(y_hat)
        confidence = float(y_proba[class_idx]) * 100
        
        probs_dict = {
            str(cls): round(float(prob) * 100, 2) 
            for cls, prob in zip(clf.classes_, y_proba)
        }
        
        results.append({
            "prediction": str(y_hat),
            "confidence_percentage": round(confidence, 2),
            "probabilities": probs_dict
        })
    
    return results