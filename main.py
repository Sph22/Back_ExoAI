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
import space_launches

# Intentar importar GCP (opcional)
try:
    import gcp_utils
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("⚠️ gcp_utils no disponible, funcionando en modo local")

DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(
    title="ExoAI API",
    version="1.1.0",
    description="API para predicción de exoplanetas con ML + Space Launches (Optimizado con GCP)"
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
    print("🚀 Iniciando ExoAI API (Optimizado)...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Inicializar GCP si está disponible
    gcp_enabled = False
    if GCP_AVAILABLE:
        gcp_enabled = gcp_utils.init_gcp()
    
    # Verificar y pre-cargar el modelo
    model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
    
    # 1. Intentar cargar modelo local
    if os.path.exists(model_path):
        print(f"✅ Modelo local encontrado en {model_path}")
        try:
            predict_model.load_bundle()
            print("✅ Modelo cargado en memoria")
        except Exception as e:
            print(f"⚠️ Error cargando modelo: {e}")
    # 2. Si no existe local, intentar descargar desde GCP
    elif gcp_enabled:
        print("📥 Modelo no encontrado localmente, intentando descargar desde GCP...")
        try:
            if gcp_utils.download_model():
                predict_model.load_bundle()
                print("✅ Modelo descargado desde GCP y cargado en memoria")
        except Exception as e:
            print(f"⚠️ No se pudo descargar modelo desde GCP: {e}")
    else:
        print(f"⚠️ Modelo NO encontrado. Ejecuta POST /train o descarga datos primero.")
    
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
        "features": [
            "Predicción de exoplanetas con ML",
            "Caché de datasets con Google Cloud Storage",
            "Próximos lanzamientos espaciales con caché local",
            "Modelo persistente en memoria (no recarga en cada request)",
            "Detección automática de misiones de exoplanetas"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "datasets": "/datasets",
            "train": "/train",
            "predict": "/predict",
            "launches": "/launches/upcoming?limit=20",
            "exoplanet_launches": "/launches/upcoming?exoplanets_only=true&limit=10"
        }
    }

@app.get("/health", tags=["Info"])
def health_check():
    """Health check para monitoreo"""
    model_exists = os.path.exists(os.getenv("MODEL_PATH", "exoplanet_model.pkl"))
    data_exists = os.path.exists(os.path.join(DATA_DIR, "cumulative.csv"))
    model_in_memory = predict_model._model_bundle is not None
    
    gcp_status = "enabled" if GCP_AVAILABLE and gcp_utils.is_initialized() else "disabled"
    
    return {
        "status": "healthy",
        "model_ready": model_exists,
        "model_in_memory": model_in_memory,
        "data_available": data_exists,
        "gcp_status": gcp_status,
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
    
    El modelo está cargado en memoria para respuestas rápidas.
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
            "model_in_memory": predict_model._model_bundle is not None,
            "test_prediction": pred,
            "test_data": test_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ---------- datasets optimizados con GCP ----------
@app.get("/datasets", tags=["Data"])
def datasets(
    auto_retrain: bool = Query(False, description="Entrena automáticamente si hay cambios"),
    token: str | None = Query(None, description="Token para autorizar auto_retrain"),
    use_cloud: bool = Query(True, description="Intentar usar datos desde GCP primero")
):
    """
    Descarga y actualiza los datasets de NASA Exoplanet Archive
    
    OPTIMIZACIÓN: Si use_cloud=True y GCP está habilitado, primero intenta
    usar datos cacheados en Cloud Storage antes de descargar desde NASA.
    """
    try:
        # Si GCP está habilitado y use_cloud=True, intentar descargar datasets existentes
        if use_cloud and GCP_AVAILABLE and gcp_utils.is_initialized():
            print("🔍 Verificando datos en Google Cloud Storage...")
            datasets_downloaded = []
            
            for dataset_name in ["cumulative", "TOI", "k2pandc"]:
                local_path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
                
                # Solo descargar si no existe localmente
                if not os.path.exists(local_path):
                    try:
                        if gcp_utils.download_dataset(dataset_name, local_path):
                            datasets_downloaded.append(dataset_name)
                            print(f"✅ {dataset_name}.csv descargado desde GCP")
                    except Exception as e:
                        print(f"⚠️ No se pudo descargar {dataset_name} desde GCP: {e}")
            
            # Si se descargaron todos los datasets desde GCP, no necesitamos fetch
            all_exist_locally = all(
                os.path.exists(os.path.join(DATA_DIR, f"{k}.csv"))
                for k in ["cumulative", "TOI", "k2pandc"]
            )
            
            if all_exist_locally:
                print("✅ Todos los datasets disponibles (desde GCP o local)")
                return JSONResponse({
                    "status": "ok",
                    "message": "Datasets cargados desde cache (GCP o local)",
                    "datasets": {
                        "cumulative": {"source": "gcp_or_local"},
                        "TOI": {"source": "gcp_or_local"},
                        "k2pandc": {"source": "gcp_or_local"}
                    },
                    "downloaded_from_gcp": datasets_downloaded
                }, status_code=200)
        
        # Si no hay datos en GCP o use_cloud=False, fetch desde NASA
        print("📡 Descargando datos desde NASA Exoplanet Archive...")
        meta = fetch_datasets.fetch_all(auto_save=True)
        
        # Subir a GCP si está habilitado
        if GCP_AVAILABLE and gcp_utils.is_initialized():
            for key in ["cumulative", "TOI", "k2pandc"]:
                path = os.path.join(DATA_DIR, f"{key}.csv")
                if os.path.exists(path):
                    try:
                        gcp_utils.upload_dataset(path, key)
                    except Exception as e:
                        print(f"⚠️ Error subiendo {key} a GCP: {e}")
        
        # Autorización de retrain
        secret = os.getenv("TRAIN_TOKEN")
        allowed = True if not secret else (token == secret)
        
        if auto_retrain and meta.get("change_detected") and allowed:
            from threading import Thread
            def _bg():
                try:
                    print("🔄 Iniciando entrenamiento automático...")
                    result = train_model.train()
                    
                    # Recargar modelo y subir a GCP
                    predict_model.reload_model()
                    
                    if GCP_AVAILABLE and gcp_utils.is_initialized():
                        model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
                        gcp_utils.upload_model(model_path, version=result.get("timestamp"))
                        gcp_utils.save_training_metrics(result)
                    
                    print("✅ Entrenamiento completado")
                except Exception as e:
                    print(f"❌ Entrenamiento falló: {e}")
            Thread(target=_bg, daemon=True).start()
            meta["retrain_started"] = True
        else:
            meta["retrain_started"] = False
        
        return JSONResponse(js(meta), status_code=200)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener datasets: {str(e)}"
        )

@app.get("/datasets/preview", tags=["Data"])
def datasets_preview(n: int = Query(5, ge=1, le=50, description="Número de filas a mostrar")):
    """Muestra las primeras N filas de cada dataset guardado"""
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
                    "solution": "Ejecuta GET /datasets para descargar los datos"
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

@app.get("/datasets/download", tags=["Data"])
def download(name: str = Query(..., pattern="^(cumulative|TOI|k2pandc)$")):
    """Descarga un dataset específico en formato CSV"""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No existe {name}.csv. Ejecuta GET /datasets primero."
        )
    return FileResponse(path, filename=f"{name}.csv", media_type="text/csv")

# ---------- entrenamiento ----------
@app.get("/train/status", tags=["ML"])
async def train_status():
    """Verifica si hay datos disponibles para entrenar"""
    cumulative_path = os.path.join(DATA_DIR, "cumulative.csv")
    model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
    
    data_exists = os.path.exists(cumulative_path)
    model_exists = os.path.exists(model_path)
    model_in_memory = predict_model._model_bundle is not None
    
    status = {
        "data_available": data_exists,
        "data_path": cumulative_path if data_exists else None,
        "model_exists": model_exists,
        "model_in_memory": model_in_memory,
        "model_path": model_path if model_exists else None,
        "ready_to_train": data_exists,
        "can_predict": model_exists and model_in_memory,
        "train_token_required": os.getenv("TRAIN_TOKEN") is not None
    }
    
    if data_exists:
        try:
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
    elif not model_in_memory:
        status["message"] = "Modelo existe pero no está en memoria. Reinicia o haz una predicción"
    else:
        status["message"] = "Todo listo. Puedes hacer predicciones en POST /predict"
    
    return status

@app.post("/train", tags=["ML"])
async def train_endpoint(token: str = Query(None, description="Token de autorización")):
    """
    Entrena el modelo de ML con los datos disponibles
    
    Después del entrenamiento, el modelo se recarga automáticamente en memoria
    y se sube a GCP si está disponible.
    """
    secret = os.getenv("TRAIN_TOKEN")
    
    if secret:
        if not token:
            raise HTTPException(status_code=403, detail="Token requerido")
        if token != secret:
            raise HTTPException(status_code=403, detail="Token inválido")
    
    try:
        cumulative_path = os.path.join(DATA_DIR, "cumulative.csv")
        if not os.path.exists(cumulative_path):
            raise HTTPException(
                status_code=400,
                detail="No hay datos para entrenar. Ejecuta GET /datasets primero."
            )
        
        print(f"🎯 Iniciando entrenamiento con datos desde: {cumulative_path}")
        out = train_model.train()
        
        # Recargar modelo en memoria
        try:
            predict_model.reload_model()
            out["model_reloaded"] = True
        except Exception as e:
            out["model_reload_error"] = str(e)
            out["model_reloaded"] = False
        
        # Subir a GCP
        if GCP_AVAILABLE and gcp_utils.is_initialized():
            model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
            version = out.get("timestamp", "manual")
            gcp_utils.upload_model(model_path, version=version)
            gcp_utils.save_training_metrics(out)
        
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
        raise HTTPException(status_code=500, detail=error_detail)

# ========== ENDPOINTS DE SPACE LAUNCHES ==========
@app.get("/launches/upcoming", tags=["Space Launches"])
async def get_upcoming_launches(
    limit: int = Query(20, ge=1, le=100, description="Número de lanzamientos"),
    force_refresh: bool = Query(False, description="Ignorar caché y actualizar"),
    exoplanets_only: bool = Query(False, description="Solo misiones de exoplanetas")
):
    """
    Obtiene próximos lanzamientos espaciales desde The Space Devs API
    
    OPTIMIZACIÓN: Los datos se cachean por 30 minutos en disco.
    Misiones de exoplanetas se detectan automáticamente y se priorizan.
    
    Respuesta incluye:
    - Información completa del lanzamiento
    - Status y probabilidad
    - Cuenta regresiva calculada
    - Flag de misión de exoplaneta
    - Proveedor, cohete y ubicación
    - URLs de info y streams
    """
    try:
        if exoplanets_only:
            data = space_launches.get_exoplanet_missions_only(limit=limit)
        else:
            data = space_launches.fetch_upcoming_launches(
                limit=limit,
                force_refresh=force_refresh
            )
        
        return JSONResponse(js(data), status_code=200)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error obteniendo lanzamientos: {str(e)}"
        )

@app.get("/launches/{launch_id}", tags=["Space Launches"])
async def get_launch_details(launch_id: str):
    """Obtiene detalles de un lanzamiento específico por ID"""
    launch = space_launches.fetch_launch_by_id(launch_id)
    
    if not launch:
        raise HTTPException(
            status_code=404,
            detail=f"Lanzamiento '{launch_id}' no encontrado"
        )
    
    return JSONResponse(js(launch), status_code=200)

@app.get("/launches/search", tags=["Space Launches"])
async def search_launches_endpoint(
    q: str = Query(..., min_length=2, description="Término de búsqueda"),
    limit: int = Query(10, ge=1, le=50)
):
    """Busca lanzamientos por nombre o misión"""
    try:
        results = space_launches.search_launches(search_query=q, limit=limit)
        return {"query": q, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda: {str(e)}")

@app.get("/launches/stats", tags=["Space Launches"])
async def get_launch_statistics():
    """
    Obtiene estadísticas de los próximos lanzamientos
    
    Incluye desglose por status, top proveedores y cantidad de misiones de exoplanetas
    """
    try:
        stats = space_launches.get_launch_statistics()
        return JSONResponse(js(stats), status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")

@app.post("/cache/clear", tags=["Utils"])
async def clear_all_caches(token: str = Query(None, description="Token de autorización")):
    """Limpia todos los cachés de la aplicación"""
    secret = os.getenv("TRAIN_TOKEN")
    if secret and token != secret:
        raise HTTPException(status_code=403, detail="Token inválido")
    
    try:
        caches_cleared = []
        
        if hasattr(fetch_datasets, '_cached_read_csv'):
            fetch_datasets._cached_read_csv.cache_clear()
            caches_cleared.append("datasets")
        
        if space_launches.clear_cache():
            caches_cleared.append("space_launches")
        
        if GCP_AVAILABLE and gcp_utils.is_initialized():
            gcp_utils.clear_cache()
            caches_cleared.append("gcp_storage")
        
        return {
            "status": "ok",
            "message": "Todos los cachés limpiados",
            "caches_cleared": caches_cleared
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- local ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    )