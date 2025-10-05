import os
from typing import List, Union

from fastapi import BackgroundTasks, FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
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

# -------- Datasets: descarga + autoentrenamiento opcional --------
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

    return JSONResponse(meta, status_code=200)

# -------- Preview seguro (sin 500) --------
@app.get("/datasets/preview")
def preview_datasets_endpoint(n: int = Query(5, ge=1, le=50)):
    try:
        data = fetch_datasets.preview(n=n)
        return JSONResponse(data, status_code=200)
    except Exception as e:
        # Nunca 500: devolvemos el error como JSON 200 para depurar sin tumbar el servicio
        return JSONResponse({"error": f"preview_failed: {type(e).__name__}: {e}"}, status_code=200)

# -------- Descarga de CSV guardado (para inspección) --------
@app.get("/datasets/download")
def download_dataset(name: str = Query(..., pattern="^(cumulative|TOI|k2pandc)$")):
    path = os.path.join(os.getenv("DATA_DIR", "data"), f"{name}.csv")
    if not os.path.exists(path):
        return JSONResponse({"error": f"No existe {path}. Ejecuta /datasets antes."}, status_code=404)
    return FileResponse(path, filename=f"{name}.csv", media_type="text/csv")

# -------- Lanzador local --------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)