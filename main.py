import os
from typing import List, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, ValidationError

import predict_model
import fetch_datasets
import train_model

app = FastAPI(title="ExoAI API")

# ===== Esquema de entrada para /predict =====
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

# ----- Predicción -----
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
        raise HTTPException(status_code=422, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=500, detail=f"Modelo no encontrado: {fe}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----- Datasets: descargar/actualizar -----
@app.get("/datasets")
def fetch_datasets_endpoint(
    auto_retrain: bool = Query(False, description="Si hay cambios, dispara entrenamiento en background"),
    token: str | None = Query(None, description="Token para habilitar auto_retrain si está protegido"),
):
    meta = fetch_datasets.fetch_all(auto_save=True)

    # Protección opcional con TRAIN_TOKEN
    secret = os.getenv("TRAIN_TOKEN")
    allow = True if not secret else (token == secret)

    if auto_retrain and meta.get("change_detected") and allow:
        # Entrena en background llamando a la función Python (no subprocess)
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

    return meta

# ----- Datasets: preview -----
@app.get("/datasets/preview")
def preview_datasets_endpoint(n: int = Query(5, ge=1, le=50)):
    try:
        return fetch_datasets.preview(n=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----- Disparar retraining explícito -----
@app.post("/retrain")
def retrain_endpoint(background_tasks: BackgroundTasks, token: str | None = Query(None)):
    secret = os.getenv("TRAIN_TOKEN")
    if secret and token != secret:
        raise HTTPException(status_code=403, detail="Token inválido")

    # Ejecuta entrenamiento en background (misma función)
    background_tasks.add_task(train_model.train)
    return {"status": "training_started"}

# ----- dev runner local -----
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)