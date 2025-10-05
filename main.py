import os
from typing import List, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

import predict_model

app = FastAPI(title="ExoAI API")

# Esquema de entrada
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
        raise HTTPException(status_code=422, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(status_code=500, detail=f"Modelo no encontrado: {fe}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Para ejecución local. En Render usa el start command (ver abajo).
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)