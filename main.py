import os
from typing import List, Union

from fastapi import BackgroundTasks, FastAPI, Query
from pydantic import BaseModel, ValidationError

import predict_model
import fetch_datasets
import train_model

app = FastAPI(title="ExoAI API")

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
        return {"error": str(ve)}
    except FileNotFoundError as fe:
        return {"error": f"Modelo no encontrado: {fe}"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/datasets")
def fetch_datasets_endpoint(
    auto_retrain: bool = Query(False, description="Si hay cambios, dispara entrenamiento en background"),
    token: str | None = Query(None, description="Token para habilitar auto_retrain si está protegido"),
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

    return meta

@app.get("/datasets/preview")
def preview_datasets_endpoint(n: int = Query(5, ge=1, le=50)):
    # Nunca devolvemos 500: si algo falla, devolvemos un JSON con el error.
    try:
        return fetch_datasets.preview(n=n)
    except Exception as e:
        return {"error": f"preview_failed: {type(e).__name__}: {e}"}

@app.post("/retrain")
def retrain_endpoint(background_tasks: BackgroundTasks, token: str | None = Query(None)):
    secret = os.getenv("TRAIN_TOKEN")
    if secret and token != secret:
        return {"error": "Token inválido"}
    background_tasks.add_task(train_model.train)
    return {"status": "training_started"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)