import os
import math
from typing import List, Union, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, ValidationError

# ======= módulos locales =======
# Asegúrate de tener estos archivos en el repo:
# - fetch_datasets.py (con funciones fetch_all() y preview())
# - predict_model.py  (con predict_one() y predict_batch())
# - train_model.py    (con train())
import fetch_datasets
import predict_model
import train_model

app = FastAPI(title="ExoAI API", version="1.0.0")

# ---------- util: sanitizar NaN/Inf para JSON ----------
def _js(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _js(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_js(v) for v in obj]
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

# ---------- modelos ----------
class InputData(BaseModel):
    period: float
    duration: float
    depth: float
    radius: float
    insolation: float
    teff: float
    srad: float

# ---------- rutas de info ----------
@app.get("/")
def root():
    return {"status": "ok", "version": app.version}

@app.get("/routes")
def routes():
    r = []
    for rt in app.router.routes:
        try:
            r.append({
                "path": rt.path,
                "name": getattr(rt, "name", None),
                "methods": list(rt.methods or [])
            })
        except Exception:
            pass
    return {"routes": r}

# ---------- predicción ----------
@app.post("/predict")
def predict_endpoint(payload: Union[InputData, List[InputData]]):
    try:
        if isinstance(payload, list):
            data = [p.model_dump() for p in payload]
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

# ---------- datasets: descarga/actualización ----------
@app.get("/datasets")
def datasets(auto_retrain: bool = Query(False),
             token: str | None = Query(None, description="protección para auto_retrain")):
    meta = fetch_datasets.fetch_all(auto_save=True)

    # Autorización de retrain
    secret = os.getenv("TRAIN_TOKEN")
    allowed = True if not secret else (token == secret)

    if auto_retrain and meta.get("change_detected") and allowed:
        from threading import Thread
        def _bg():
            try:
                train_model.train()
            except Exception as e:
                print("Entrenamiento falló:", e)
        Thread(target=_bg, daemon=True).start()
        meta["retrain_started"] = True
    else:
        meta["retrain_started"] = False

    return JSONResponse(_js(meta), status_code=200)

# ---------- preview de CSVs guardados ----------
@app.get("/datasets/preview")
def datasets_preview(n: int = Query(5, ge=1, le=50)):
    try:
        data = fetch_datasets.preview(n=n)
        return JSONResponse(_js(data), status_code=200)
    except Exception as e:
        return JSONResponse(_js({"error": f"preview_failed: {type(e).__name__}: {e}"}), status_code=200)

# ---------- descargar un CSV (último guardado) ----------
@app.get("/datasets/download")
def download(name: str = Query(..., pattern="^(cumulative|TOI|k2pandc)$")):
    data_dir = os.getenv("DATA_DIR", "data")
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        return JSONResponse({"error": f"No existe {path}. Ejecuta /datasets primero."}, status_code=404)
    return FileResponse(path, filename=f"{name}.csv", media_type="text/csv")

# ---------- entrenamiento manual ----------
@app.post("/train")
def train(token: str = Query(None)):
    secret = os.getenv("TRAIN_TOKEN")
    if secret and token != secret:
        return JSONResponse({"error": "Token inválido"}, status_code=403)
    try:
        out = train_model.train()
        return JSONResponse(_js(out), status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- local ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))