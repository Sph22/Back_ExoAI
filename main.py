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
import gcp_utils  # NUEVO: reemplaza firebase_utils

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(
    title="ExoAI API",
    version="1.0.4",
    description="API para predicción de exoplanetas con ML + Google Cloud"
)

# ========== CONFIGURACIÓN DE CORS ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    
    # Inicializar Google Cloud
    gcp_initialized = gcp_utils.init_gcp()
    
    # Verificar si existe el modelo
    model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
    if os.path.exists(model_path):
        print(f"✅ Modelo local encontrado en {model_path}")
    else:
        print(f"⚠️ Modelo local NO encontrado en {model_path}")
        # Intentar descargar desde GCP
        if gcp_initialized:
            print("🔄 Intentando descargar modelo desde Google Cloud...")
            downloaded = gcp_utils.download_model()
            if downloaded:
                print(f"✅ Modelo descargado desde GCP")
            else:
                print("❌ No se pudo descargar modelo desde GCP")
    
    # Listar rutas disponibles
    rutas = []
    for rt in app.router.routes:
        if hasattr(rt, 'path'):
            methods = list(getattr(rt, 'methods', []))
            rutas.append(f"{' '.join(methods):6} {rt.path}")
    
    print("\n📋 RUTAS DISPONIBLES:")
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
            "predict": "/predict",
            "predictions_history": "/predictions/history",
            "cloud_storage": "/cloud/storage/list"
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
        "data_dir": DATA_DIR,
        "gcp_enabled": gcp_utils.is_initialized()
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
async def predict_endpoint(
    payload: Union[InputData, List[InputData]],
    save_to_cloud: bool = Query(False, description="Guardar predicción en Firestore")
):
    """
    Predice si un objeto es un exoplaneta confirmado
    
    - **payload**: Datos del objeto (individual o lista)
    - **save_to_cloud**: Guardar resultado en Firestore (GCP)
    
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
    
    Respuesta:
    ```json
    {
        "prediction": "CANDIDATE",
        "confidence_percentage": 87.45,
        "probabilities": {
            "CANDIDATE": 87.45,
            "CONFIRMED": 8.32,
            "FALSE POSITIVE": 4.23
        }
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
            
            # Guardar en Firestore si está habilitado
            if save_to_cloud and gcp_utils.is_initialized():
                doc_id = gcp_utils.save_prediction(data, pred)
                if doc_id:
                    pred["firestore_id"] = doc_id
            
            return pred
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fe:
        raise HTTPException(
            status_code=404,
            detail=f"Modelo no encontrado. Ejecuta POST /train primero. Error: {fe}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/history", tags=["ML"])
async def get_predictions_history(limit: int = Query(100, ge=1, le=1000)):
    """Obtiene historial de predicciones desde Firestore"""
    if not gcp_utils.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Google Cloud no está inicializado"
        )
    
    predictions = gcp_utils.get_predictions(limit=limit)
    return {"predictions": predictions, "count": len(predictions)}

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
        
        # Subir datasets a Google Cloud Storage
        if gcp_utils.is_initialized():
            for key in ["cumulative", "TOI", "k2pandc"]:
                path = os.path.join(DATA_DIR, f"{key}.csv")
                if os.path.exists(path):
                    gcp_utils.upload_dataset(path, key)
        
        # Autorización de retrain
        secret = os.getenv("TRAIN_TOKEN")
        allowed = True if not secret else (token == secret)
        
        if auto_retrain and meta.get("change_detected") and allowed:
            from threading import Thread
            def _bg():
                try:
                    print("Iniciando entrenamiento automático...")
                    train_result = train_model.train()
                    
                    # Subir modelo y métricas a GCP
                    if gcp_utils.is_initialized():
                        model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
                        version = train_result.get("timestamp", "auto")
                        gcp_utils.upload_model(model_path, version=version)
                        gcp_utils.save_training_metrics(train_result)
                    
                    print("Entrenamiento completado")
                except Exception as e:
                    print(f"Entrenamiento falló: {e}")
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
    
    if secret:
        if not token:
            raise HTTPException(
                status_code=403, 
                detail="Token requerido. TRAIN_TOKEN está configurado."
            )
        if token != secret:
            raise HTTPException(
                status_code=403, 
                detail="Token inválido"
            )
    
    try:
        cumulative_path = os.path.join(DATA_DIR, "cumulative.csv")
        if not os.path.exists(cumulative_path):
            raise HTTPException(
                status_code=400,
                detail="No hay datos para entrenar. Ejecuta GET /datasets primero."
            )
        
        print(f"Iniciando entrenamiento con datos desde: {cumulative_path}")
        out = train_model.train()
        
        # Subir modelo y métricas a GCP
        if gcp_utils.is_initialized():
            model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
            version = out.get("timestamp", "manual")
            model_url = gcp_utils.upload_model(model_path, version=version)
            gcp_utils.save_training_metrics(out)
            if model_url:
                out["gcp_model_path"] = model_url
        
        print(f"Entrenamiento completado: {out}")
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
        print(f"Error en /train: {error_detail}")
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
        "train_token_required": os.getenv("TRAIN_TOKEN") is not None,
        "gcp_enabled": gcp_utils.is_initialized()
    }
    
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
    
    # Listar modelos en GCP Storage
    if gcp_utils.is_initialized():
        status["gcp_model_versions"] = gcp_utils.list_model_versions()
    
    if not data_exists:
        status["message"] = "Ejecuta GET /datasets para descargar los datos primero"
    elif not model_exists:
        status["message"] = "Datos disponibles. Ejecuta POST /train para entrenar el modelo"
    else:
        status["message"] = "Todo listo. Puedes hacer predicciones en POST /predict"
    
    return status

@app.get("/train/history", tags=["ML"])
async def training_history(limit: int = Query(50, ge=1, le=200)):
    """Obtiene historial de entrenamientos desde Firestore"""
    if not gcp_utils.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Google Cloud no está inicializado"
        )
    
    history = gcp_utils.get_training_history(limit=limit)
    return {"history": history, "count": len(history)}

# ---------- endpoints de Cloud Storage ----------
@app.get("/cloud/storage/list", tags=["Cloud"])
async def list_cloud_files(prefix: str = Query("", description="Prefijo para filtrar archivos")):
    """Lista archivos en Google Cloud Storage"""
    if not gcp_utils.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Google Cloud no está inicializado"
        )
    
    files = gcp_utils.list_files(prefix=prefix)
    return {"files": files, "count": len(files)}

@app.get("/cloud/models/versions", tags=["Cloud"])
async def list_model_versions():
    """Lista versiones de modelos disponibles en GCP"""
    if not gcp_utils.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Google Cloud no está inicializado"
        )
    
    versions = gcp_utils.list_model_versions()
    return {"versions": versions, "count": len(versions)}

@app.post("/cloud/models/download", tags=["Cloud"])
async def download_model_from_cloud(
    version: str = Query("latest", description="Versión a descargar"),
    token: str = Query(None, description="Token de autorización")
):
    """Descarga un modelo específico desde GCP"""
    secret = os.getenv("TRAIN_TOKEN")
    if secret and token != secret:
        raise HTTPException(status_code=403, detail="Token inválido")
    
    if not gcp_utils.is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Google Cloud no está inicializado"
        )
    
    local_path = gcp_utils.download_model(version=version)
    if local_path:
        return {"status": "ok", "message": f"Modelo descargado: {local_path}"}
    else:
        raise HTTPException(status_code=404, detail=f"No se encontró modelo versión '{version}'")

# ---------- local ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    )