import os
import math
from typing import List, Union, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

# ===== módulos locales =====
import fetch_datasets
import predict_model
import train_model

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(
    title="ExoAI API",
    version="1.0.2",
    description="API para predicción de exoplanetas con ML"
)

# ========== CONFIGURACIÓN DE CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica dominios: ["https://tu-frontend.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Permite GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Permite todos los headers
)

# ---------- util: sanitizar NaN/Inf para JSON ----------
def js(obj: Any):
    if isinstance(obj, dict):
        return {str(k): js(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [js(v) for v in obj]
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

# ---------- hooks de arranque ----------
@app.on_event("startup")
def on_startup():
    print("🚀 Iniciando ExoAI API...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Verificar si existe el modelo
    model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
    if os.path.exists(model_path):
        print(f"✅ Modelo encontrado en {model_path}")
    else:
        print(f"⚠️  Modelo NO encontrado en {model_path}. Ejecuta /train primero.")
    
    # Listar rutas disponibles
    rutas = []
    for rt in app.router.routes:
        if hasattr(rt, 'path'):
            methods = list(getattr(rt, 'methods', []))
            rutas.append(f"{' '.join(methods):6} {rt.path}")
    
    print("\n📍 RUTAS DISPONIBLES:")
    for ruta in sorted(rutas):
        print(f"   {ruta}")
    print()

# ---------- info básica ----------
@app.get("/", tags=["Info"])
def root():
    """Endpoint raíz - información básica de la API"""
    return {
        "status": "ok",
        "version": app.version,
        "title": app.title,
        "endpoints": {
            "docs": "/docs",
            "routes": "/routes",
            "health": "/health",
            "datasets": "/datasets",
            "preview": "/datasets/preview?n=5",
            "train": "/train",
            "predict": "/predict"
        }
    }

@app.get("/health", tags=["Info"])
def health_check():
    """Health check para monitoreo"""
    model_exists = os.path.exists(os.getenv("MODEL_PATH", "exoplanet_model.pkl"))
    data_exists = os.path.exists(os.path.join(DATA_DIR, "cumulative.csv"))
    
    return {
        "status": "healthy",
        "model_ready": model_exists,
        "data_available": data_exists,
        "data_dir": DATA_DIR
    }

@app.get("/routes", tags=["Info"])
def list_routes():
    """Lista todas las rutas disponibles en la API"""
    info = []
    for rt in app.router.routes:
        try:
            info.append({
                "path": rt.path,
                "name": getattr(rt, "name", None),
                "methods": list(rt.methods or [])
            })
        except Exception:
            pass
    return {"routes": info}

# ---------- predicción ----------
@app.post("/predict", tags=["ML"])
async def predict_endpoint(payload: Union[InputData, List[InputData]]):
    """
    Predice si un objeto es un exoplaneta confirmado
    
    - **payload**: Datos del objeto (individual o lista)
    
    Ejemplo de payload:
    ```json
    {
        "period": 3.52,
        "duration": 2.8,
        "depth": 4500,
        "radius": 2.5,
        "insolation": 150,
        "teff": 5800,
        "srad": 1.1
    }
    ```
    """
    try:
        if isinstance(payload, list):
            data = [p.model_dump() for p in payload]
            preds = predict_model.predict_batch(data)
            return {"predictions": preds, "count": len(preds)}
        else:
            data = payload.model_dump()
            pred = predict_model.predict_one(data)
            return {"prediction": pred}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado. Ejecuta POST /train primero. Error: {fe}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/test", tags=["ML"])
async def predict_test():
    """Endpoint de prueba para verificar que el modelo está cargado"""
    try:
        model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
        if not os.path.exists(model_path):
            return {
                "status": "error",
                "message": f"Modelo no encontrado en {model_path}",
                "solution": "Ejecuta POST /train primero"
            }
        
        # Hacer una predicción de prueba
        test_data = {
            "period": 3.52,
            "duration": 2.8,
            "depth": 4500.0,
            "radius": 2.5,
            "insolation": 150.0,
            "teff": 5800.0,
            "srad": 1.1
        }
        pred = predict_model.predict_one(test_data)
        
        return {
            "status": "ok",
            "model_loaded": True,
            "test_prediction": pred,
            "test_data": test_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ---------- datasets: descarga/actualización ----------
@app.get("/datasets", tags=["Data"])
def datasets(
    auto_retrain: bool = Query(False, description="Entrena automáticamente si hay cambios"),
    token: str | None = Query(None, description="Token para autorizar auto_retrain")
):
    """
    Descarga y actualiza los datasets de NASA Exoplanet Archive
    
    - **auto_retrain**: Si True y hay cambios, entrena el modelo automáticamente
    - **token**: Requerido si TRAIN_TOKEN está configurado
    """
    try:
        meta = fetch_datasets.fetch_all(auto_save=True)
        
        # Autorización de retrain
        secret = os.getenv("TRAIN_TOKEN")
        allowed = True if not secret else (token == secret)
        
        if auto_retrain and meta.get("change_detected") and allowed:
            from threading import Thread
            def _bg():
                try:
                    print("🔄 Iniciando entrenamiento automático...")
                    train_model.train()
                    print("✅ Entrenamiento completado")
                except Exception as e:
                    print(f"❌ Entrenamiento falló: {e}")
            Thread(target=_bg, daemon=True).start()
            meta["retrain_started"] = True
        else:
            meta["retrain_started"] = False
            if auto_retrain and not allowed:
                meta["retrain_denied"] = "Token inválido o no proporcionado"
        
        return JSONResponse(js(meta), status_code=200)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener datasets: {str(e)}"
        )

# ---------- preview de CSVs guardados ----------
@app.get("/datasets/preview", tags=["Data"])
def datasets_preview(n: int = Query(5, ge=1, le=50, description="Número de filas a mostrar")):
    """
    Muestra las primeras N filas de cada dataset guardado
    
    - **n**: Número de filas (entre 1 y 50)
    """
    try:
        # Verificar si existen los archivos
        missing_files = []
        for key in ["cumulative", "TOI", "k2pandc"]:
            path = os.path.join(DATA_DIR, f"{key}.csv")
            if not os.path.exists(path):
                missing_files.append(key)
        
        if missing_files:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Archivos no encontrados",
                    "missing": missing_files,
                    "solution": "Ejecuta GET /datasets primero para descargar los datos"
                }
            )
        
        data = fetch_datasets.preview(n=n)
        return JSONResponse(js(data), status_code=200)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al leer preview: {type(e).__name__}: {str(e)}"
        )

# ---------- descargar un CSV ----------
@app.get("/datasets/download", tags=["Data"])
def download(name: str = Query(..., pattern="^(cumulative|TOI|k2pandc)$")):
    """
    Descarga un dataset específico en formato CSV
    
    - **name**: Nombre del dataset (cumulative, TOI, o k2pandc)
    """
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No existe {name}.csv. Ejecuta GET /datasets primero."
        )
    return FileResponse(path, filename=f"{name}.csv", media_type="text/csv")

# ---------- entrenamiento manual ----------
@app.post("/train", tags=["ML"])
async def train_endpoint(token: str = Query(None, description="Token de autorización")):
    """
    Entrena el modelo de ML con los datos disponibles
    
    - **token**: Requerido si TRAIN_TOKEN está configurado como variable de entorno
    
    El entrenamiento puede tardar 1-2 minutos dependiendo del tamaño de los datos.
    """
    secret = os.getenv("TRAIN_TOKEN")
    
    # Validar token solo si está configurado
    if secret:
        if not token:
            raise HTTPException(
                status_code=403, 
                detail=f"Token requerido. TRAIN_TOKEN está configurado."
            )
        if token != secret:
            raise HTTPException(
                status_code=403, 
                detail=f"Token inválido. Recibido: '{token[:5]}...'"
            )
    
    try:
        # Verificar que existan los datos
        cumulative_path = os.path.join(DATA_DIR, "cumulative.csv")
        if not os.path.exists(cumulative_path):
            raise HTTPException(
                status_code=400,
                detail="No hay datos para entrenar. Ejecuta GET /datasets primero."
            )
        
        print(f"🎯 Iniciando entrenamiento con datos desde: {cumulative_path}")
        out = train_model.train()
        print(f"✅ Entrenamiento completado: {out}")
        
        return JSONResponse(js(out), status_code=200)
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": traceback.format_exc()
        }
        print(f"❌ Error en /train: {error_detail}")
        raise HTTPException(
            status_code=500,
            detail=error_detail
        )

@app.get("/train/status", tags=["ML"])
async def train_status():
    """Verifica si hay datos disponibles para entrenar"""
    cumulative_path = os.path.join(DATA_DIR, "cumulative.csv")
    model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
    
    data_exists = os.path.exists(cumulative_path)
    model_exists = os.path.exists(model_path)
    
    status = {
        "data_available": data_exists,
        "data_path": cumulative_path if data_exists else None,
        "model_exists": model_exists,
        "model_path": model_path if model_exists else None,
        "ready_to_train": data_exists,
        "can_predict": model_exists,
        "train_token_required": os.getenv("TRAIN_TOKEN") is not None
    }
    
    # Si hay datos, verificar contenido
    if data_exists:
        try:
            import pandas as pd
            df = pd.read_csv(cumulative_path, nrows=5)
            status["data_sample"] = {
                "rows_sample": len(df),
                "total_columns": len(df.columns),
                "columns_preview": list(df.columns[:20])
            }
        except Exception as e:
            status["data_read_error"] = str(e)
    
    if not data_exists:
        status["message"] = "Ejecuta GET /datasets para descargar los datos primero"
    elif not model_exists:
        status["message"] = "Datos disponibles. Ejecuta POST /train para entrenar el modelo"
    else:
        status["message"] = "Todo listo. Puedes hacer predicciones en POST /predict"
    
    return status

# ---------- local ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    )