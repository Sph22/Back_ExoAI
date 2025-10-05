import os
import math
from typing import List, Union, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, ValidationError

import predict_model
import fetch_datasets
import train_model

app = FastAPI(title="ExoAI API")

# ---------- JSON sanitize ----------
def json_sanitize(obj: Any):
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_sanitize(v) for v in obj]
    if isinstance(obj, (np.generic,)):
        obj = obj.item()
    try:
        if obj is None:
            return None
        if isinstance(obj, float):
            if not math.isfinite(obj):
                return None
            return float(obj)
        if isinstance(obj, (int, str, bool)):
            return obj
        if pd.isna(obj):
            return None
    except Exception:
        return None
    return str(obj)

class InputData(BaseModel):
    period: float
    duration: float
    depth: float
    radius: float
    insolation: float
    teff: float
    srad: float

@app.get("/")
def root():
    return {"status": "ok"}

# -------- Predicción --------
@app.post("/predict")
def predict_endpoint(payload: Union[InputData, List[InputData]]):
    try:
        if isinstance(payload, list):
            data = [item.model_dump() for item in payload]
            preds = predict_model.predict_batch(data)
            return {"predictions": preds}
        else:
            data = payload.model_dump()
            pred = predict_model.predict_one(data)
            return {"prediction": pred}
    except ValidationError as ve:
        return JSONResponse({"error": str(ve)}, status_code=400)
    except FileNotFoundError as fe:
        return JSONResponse({"error": f"Modelo no encontrado: {fe}"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -------- Datasets + auto-retrain --------
@app.get("/datasets")
def fetch_datasets_endpoint(
    auto_retrain: bool = Query(False),
    token: str | None = Query(None),
):
    meta = fetch_datasets.fetch_all(auto_save=True)

    secret = os.getenv("TRAIN_TOKEN")
    allow = True if not secret else (token == secret)

    if auto_retrain and meta.get("change_detected") and allow:
        def _train():
            try:
                train_model.train()
            except Exception as ex:
                print("Entrenamiento falló:", ex)
        from threading import Thread
        Thread(target=_train, daemon=True).start()
        meta["retrain_started"] = True
    else:
        meta["retrain_started"] = False

    return JSONResponse(json_sanitize(meta), status_code=200)

# -------- Preview --------
@app.get("/datasets/preview")
def preview_datasets_endpoint(n: int = Query(5, ge=1, le=50)):
    try:
        data = fetch_datasets.preview(n=n)
        return JSONResponse(json_sanitize(data), status_code=200)
    except Exception as e:
        return JSONResponse(json_sanitize({"error": f"preview_failed: {type(e).__name__}: {e}"}), status_code=200)

# -------- Descargar CSV guardado --------
@app.get("/datasets/download")
def download_dataset(name: str = Query(..., pattern="^(cumulative|TOI|k2pandc)$")):
    path = os.path.join(os.getenv("DATA_DIR", "data"), f"{name}.csv")
    if not os.path.exists(path):
        return JSONResponse({"error": f"No existe {path}. Ejecuta /datasets antes."}, status_code=404)
    return FileResponse(path, filename=f"{name}.csv", media_type="text/csv")

# -------- Entrenamiento manual --------
@app.post("/train")
def train_endpoint(token: str = Query(None)):
    secret = os.getenv("TRAIN_TOKEN")
    allow = True if not secret else (token == secret)
    if not allow:
        return JSONResponse({"error": "Token inválido"}, status_code=403)
    try:
        result = train_model.train()
        return JSONResponse(json_sanitize(result), status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -------- Local --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)