import os
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from io import BytesIO
from functools import lru_cache

from google.cloud import storage
from google.cloud import firestore
from google.oauth2 import service_account

# Variables globales
_storage_client = None
_firestore_client = None
_bucket = None
_initialized = False

def init_gcp():
    """
    Inicializa clientes de GCP usando credenciales del service account.
    Busca credenciales en:
    1. Variable de entorno GOOGLE_APPLICATION_CREDENTIALS (ruta al JSON)
    2. Variable de entorno GCP_CREDENTIALS_JSON (contenido del JSON)
    """
    global _storage_client, _firestore_client, _bucket, _initialized
    
    if _initialized:
        return True
    
    try:
        credentials = None
        project_id = os.getenv("GCP_PROJECT_ID")
        
        # Opción 1: Ruta a archivo JSON
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and os.path.exists(cred_path):
            credentials = service_account.Credentials.from_service_account_file(cred_path)
            if not project_id:
                with open(cred_path) as f:
                    project_id = json.load(f).get("project_id")
        
        # Opción 2: JSON como string en variable de entorno
        elif os.getenv("GCP_CREDENTIALS_JSON"):
            cred_json = json.loads(os.getenv("GCP_CREDENTIALS_JSON"))
            credentials = service_account.Credentials.from_service_account_info(cred_json)
            if not project_id:
                project_id = cred_json.get("project_id")
        
        if not credentials:
            print("⚠️ GCP credentials no encontradas, usando modo local")
            return False
        
        # Inicializar clientes
        _storage_client = storage.Client(credentials=credentials, project=project_id)
        _firestore_client = firestore.Client(credentials=credentials, project=project_id)
        
        # Obtener bucket
        bucket_name = os.getenv("GCP_BUCKET_NAME", "exoai-models")
        _bucket = _storage_client.bucket(bucket_name)
        
        _initialized = True
        print(f"✅ GCP inicializado - Proyecto: {project_id}, Bucket: {bucket_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error inicializando GCP: {e}")
        return False

def is_initialized() -> bool:
    """Verifica si GCP está inicializado"""
    return _initialized

# ========== CLOUD STORAGE ==========

def upload_file(local_path: str, remote_path: str, make_public: bool = False) -> Optional[str]:
    """
    Sube un archivo a Cloud Storage
    
    Args:
        local_path: Ruta local del archivo
        remote_path: Ruta en el bucket (ej: "models/model_v1.pkl")
        make_public: Si True, hace el archivo público
    
    Returns:
        URL pública si make_public=True, o ruta del blob
    """
    if not _initialized:
        print("GCP no inicializado")
        return None
    
    try:
        blob = _bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        
        if make_public:
            blob.make_public()
            url = blob.public_url
            print(f"✅ Archivo subido (público): {url}")
            return url
        else:
            print(f"✅ Archivo subido: gs://{_bucket.name}/{remote_path}")
            return f"gs://{_bucket.name}/{remote_path}"
            
    except Exception as e:
        print(f"❌ Error subiendo archivo: {e}")
        return None

def download_file(remote_path: str, local_path: str) -> bool:
    """
    Descarga un archivo de Cloud Storage
    
    Args:
        remote_path: Ruta en el bucket
        local_path: Ruta local donde guardar
    
    Returns:
        True si exitoso, False si falló
    """
    if not _initialized:
        return False
    
    try:
        blob = _bucket.blob(remote_path)
        blob.download_to_filename(local_path)
        print(f"✅ Archivo descargado a: {local_path}")
        return True
    except Exception as e:
        print(f"❌ Error descargando archivo: {e}")
        return False

def download_file_bytes(remote_path: str) -> Optional[bytes]:
    """
    Descarga un archivo como bytes directamente sin guardar en disco
    
    Args:
        remote_path: Ruta en el bucket
    
    Returns:
        Contenido del archivo en bytes o None si falla
    """
    if not _initialized:
        return None
    
    try:
        blob = _bucket.blob(remote_path)
        return blob.download_as_bytes()
    except Exception as e:
        print(f"❌ Error descargando bytes: {e}")
        return None

def upload_bytes(data: bytes, remote_path: str, content_type: str = None) -> Optional[str]:
    """
    Sube bytes directamente a Cloud Storage
    
    Args:
        data: Datos en bytes
        remote_path: Ruta en el bucket
        content_type: MIME type (ej: "application/octet-stream")
    
    Returns:
        Ruta del blob
    """
    if not _initialized:
        return None
    
    try:
        blob = _bucket.blob(remote_path)
        blob.upload_from_file(BytesIO(data), content_type=content_type)
        print(f"✅ Bytes subidos: gs://{_bucket.name}/{remote_path}")
        return f"gs://{_bucket.name}/{remote_path}"
    except Exception as e:
        print(f"❌ Error subiendo bytes: {e}")
        return None

@lru_cache(maxsize=10)
def list_files(prefix: str = "") -> tuple:
    """
    Lista archivos en el bucket con un prefijo dado (con caché)
    
    Args:
        prefix: Prefijo para filtrar (ej: "models/")
    
    Returns:
        Tupla de nombres de archivos (inmutable para caché)
    """
    if not _initialized:
        return tuple()
    
    try:
        blobs = _bucket.list_blobs(prefix=prefix)
        return tuple(blob.name for blob in blobs)
    except Exception as e:
        print(f"❌ Error listando archivos: {e}")
        return tuple()

def delete_file(remote_path: str) -> bool:
    """Elimina un archivo del bucket"""
    if not _initialized:
        return False
    
    try:
        blob = _bucket.blob(remote_path)
        blob.delete()
        print(f"✅ Archivo eliminado: {remote_path}")
        # Limpiar caché
        list_files.cache_clear()
        return True
    except Exception as e:
        print(f"❌ Error eliminando archivo: {e}")
        return False

@lru_cache(maxsize=100)
def file_exists(remote_path: str) -> bool:
    """Verifica si un archivo existe en el bucket (con caché)"""
    if not _initialized:
        return False
    
    try:
        blob = _bucket.blob(remote_path)
        return blob.exists()
    except Exception as e:
        return False

# ========== FIRESTORE ==========

def save_prediction(data: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
    """
    Guarda una predicción en Firestore
    
    Args:
        data: Datos de entrada
        result: Resultado de la predicción
    
    Returns:
        ID del documento creado
    """
    if not _initialized:
        return None
    
    try:
        doc_ref = _firestore_client.collection('predictions').add({
            'input': data,
            'output': result,
            'timestamp': datetime.utcnow(),
            'model_version': os.getenv("MODEL_VERSION", "1.0")
        })
        doc_id = doc_ref[1].id
        print(f"✅ Predicción guardada: {doc_id}")
        return doc_id
    except Exception as e:
        print(f"❌ Error guardando predicción: {e}")
        return None

def get_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Obtiene últimas predicciones de Firestore
    
    Args:
        limit: Número máximo de predicciones a obtener
    
    Returns:
        Lista de predicciones con sus IDs
    """
    if not _initialized:
        return []
    
    try:
        docs = _firestore_client.collection('predictions')\
                                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                                .limit(limit)\
                                .stream()
        
        results = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            results.append(data)
        
        return results
    except Exception as e:
        print(f"❌ Error obteniendo predicciones: {e}")
        return []

def save_training_metrics(metrics: Dict[str, Any]) -> Optional[str]:
    """
    Guarda métricas de entrenamiento en Firestore
    
    Args:
        metrics: Diccionario con métricas (accuracy, report, etc.)
    
    Returns:
        ID del documento creado
    """
    if not _initialized:
        return None
    
    try:
        doc_ref = _firestore_client.collection('training_history').add({
            'metrics': metrics,
            'timestamp': datetime.utcnow(),
            'model_path': os.getenv("MODEL_PATH", "exoplanet_model.pkl")
        })
        doc_id = doc_ref[1].id
        print(f"✅ Métricas guardadas: {doc_id}")
        return doc_id
    except Exception as e:
        print(f"❌ Error guardando métricas: {e}")
        return None

def get_training_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Obtiene historial de entrenamientos"""
    if not _initialized:
        return []
    
    try:
        docs = _firestore_client.collection('training_history')\
                                .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                                .limit(limit)\
                                .stream()
        
        results = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            results.append(data)
        
        return results
    except Exception as e:
        print(f"❌ Error obteniendo historial: {e}")
        return []

# ========== HELPERS PARA MODELOS ==========

def upload_model(local_path: str, version: str = None) -> Optional[str]:
    """
    Sube un modelo a Cloud Storage con versionado
    
    Args:
        local_path: Ruta local del modelo
        version: Versión del modelo (si None, usa timestamp)
    
    Returns:
        Ruta remota del modelo
    """
    if not version:
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    filename = os.path.basename(local_path)
    remote_path = f"models/{version}/{filename}"
    
    result = upload_file(local_path, remote_path)
    
    # También subir como "latest"
    if result:
        upload_file(local_path, f"models/latest/{filename}")
        # Limpiar caché
        list_files.cache_clear()
        file_exists.cache_clear()
    
    return result

def download_model(version: str = "latest", local_path: str = None) -> Optional[str]:
    """
    Descarga un modelo desde Cloud Storage
    
    Args:
        version: Versión a descargar ("latest" por defecto)
        local_path: Ruta local (si None, usa MODEL_PATH)
    
    Returns:
        Ruta local del archivo descargado
    """
    if not local_path:
        local_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
    
    filename = os.path.basename(local_path)
    remote_path = f"models/{version}/{filename}"
    
    if download_file(remote_path, local_path):
        return local_path
    return None

def list_model_versions() -> List[str]:
    """Lista todas las versiones de modelos disponibles"""
    files = list_files(prefix="models/")
    versions = set()
    
    for file_path in files:
        parts = file_path.split("/")
        if len(parts) >= 2:
            versions.add(parts[1])  # models/VERSION/...
    
    return sorted(list(versions))

# ========== HELPERS PARA DATASETS ==========

def upload_dataset(local_path: str, dataset_name: str) -> Optional[str]:
    """
    Sube un dataset a Cloud Storage
    
    Args:
        local_path: Ruta local del CSV
        dataset_name: Nombre del dataset (cumulative, TOI, k2pandc)
    
    Returns:
        Ruta remota del dataset
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    remote_path = f"datasets/{dataset_name}_{timestamp}.csv"
    
    result = upload_file(local_path, remote_path)
    
    # También subir como "latest"
    if result:
        upload_file(local_path, f"datasets/{dataset_name}_latest.csv")
        list_files.cache_clear()
    
    return result

def download_dataset(dataset_name: str, local_path: str = None) -> Optional[str]:
    """Descarga la última versión de un dataset"""
    if not local_path:
        local_path = os.path.join(os.getenv("DATA_DIR", "data"), f"{dataset_name}.csv")
    
    remote_path = f"datasets/{dataset_name}_latest.csv"
    
    if download_file(remote_path, local_path):
        return local_path
    return None

def clear_cache():
    """Limpia todos los cachés de lru_cache"""
    list_files.cache_clear()
    file_exists.cache_clear()
    print("✅ Caché de GCP limpiado")