"""
Sistema de actualización automática de datasets y modelo.
Se ejecuta periódicamente para mantener datos actualizados en GCP.
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict
import hashlib

import fetch_datasets
import train_model
import predict_model

try:
    import gcp_utils
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

DATA_DIR = os.getenv("DATA_DIR", "data")

def get_dataset_hash(dataset_name: str) -> str:
    """Calcula hash SHA256 de un dataset local"""
    path = os.path.join(DATA_DIR, f"{dataset_name}.csv")
    if not os.path.exists(path):
        return ""
    
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def check_for_updates() -> Dict[str, bool]:
    """
    Verifica si hay actualizaciones en los datasets de NASA.
    Retorna dict con cambios detectados por dataset.
    """
    print(f"\n{'='*60}")
    print(f"🔍 Verificando actualizaciones - {datetime.utcnow().isoformat()}")
    print(f"{'='*60}")
    
    # Guardar hashes actuales
    old_hashes = {
        "cumulative": get_dataset_hash("cumulative"),
        "TOI": get_dataset_hash("TOI"),
        "k2pandc": get_dataset_hash("k2pandc")
    }
    
    try:
        # Descargar datasets desde NASA
        print("\n📡 Descargando datasets desde NASA Exoplanet Archive...")
        result = fetch_datasets.fetch_all(auto_save=True)
        
        # Comparar hashes
        changes = {}
        for dataset in ["cumulative", "TOI", "k2pandc"]:
            new_hash = get_dataset_hash(dataset)
            changed = old_hashes[dataset] != new_hash and new_hash != ""
            changes[dataset] = changed
            
            if changed:
                print(f"✅ {dataset}: ACTUALIZADO")
            else:
                print(f"⚪ {dataset}: Sin cambios")
        
        return {
            "any_changes": any(changes.values()),
            "changes": changes,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Error verificando actualizaciones: {e}")
        return {
            "any_changes": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

def upload_to_gcp() -> bool:
    """Sube datasets actualizados a GCP"""
    if not GCP_AVAILABLE or not gcp_utils.is_initialized():
        print("⚠️ GCP no disponible, saltando upload")
        return False
    
    print("\n☁️ Subiendo datasets a Google Cloud Storage...")
    
    try:
        for dataset in ["cumulative", "TOI", "k2pandc"]:
            path = os.path.join(DATA_DIR, f"{dataset}.csv")
            if os.path.exists(path):
                gcp_utils.upload_dataset(path, dataset)
                print(f"✅ {dataset} subido a GCP")
        
        return True
    except Exception as e:
        print(f"❌ Error subiendo a GCP: {e}")
        return False

def retrain_model() -> Dict:
    """Re-entrena el modelo con datos actualizados"""
    print("\n🎯 Iniciando re-entrenamiento del modelo...")
    
    try:
        result = train_model.train()
        print(f"✅ Modelo entrenado - Accuracy: {result['accuracy']:.4f}")
        
        # Recargar modelo en memoria
        try:
            predict_model.reload_model()
            print("✅ Modelo recargado en memoria")
        except Exception as e:
            print(f"⚠️ Error recargando modelo: {e}")
        
        # Subir modelo a GCP
        if GCP_AVAILABLE and gcp_utils.is_initialized():
            model_path = os.getenv("MODEL_PATH", "exoplanet_model.pkl")
            version = result.get("timestamp", "auto")
            gcp_utils.upload_model(model_path, version=version)
            gcp_utils.save_training_metrics(result)
            print(f"✅ Modelo v{version} subido a GCP")
        
        return result
        
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        return {"error": str(e)}

async def auto_update_cycle():
    """Ciclo completo de actualización automática"""
    print("\n" + "="*60)
    print("🤖 INICIANDO CICLO DE ACTUALIZACIÓN AUTOMÁTICA")
    print("="*60)
    
    # 1. Verificar actualizaciones
    update_status = check_for_updates()
    
    if update_status.get("any_changes"):
        print("\n🔄 Cambios detectados, procediendo con actualización...")
        
        # 2. Subir a GCP
        upload_to_gcp()
        
        # 3. Re-entrenar modelo
        train_result = retrain_model()
        
        print("\n" + "="*60)
        print("✅ ACTUALIZACIÓN COMPLETADA")
        print(f"Accuracy: {train_result.get('accuracy', 'N/A')}")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        print("="*60 + "\n")
        
        return {
            "status": "updated",
            "update_status": update_status,
            "train_result": train_result
        }
    else:
        print("\n✓ No hay cambios, datasets ya están actualizados")
        print("="*60 + "\n")
        return {
            "status": "no_changes",
            "update_status": update_status
        }

def force_full_update():
    """Fuerza actualización completa (para uso manual)"""
    print("\n🔥 FORZANDO ACTUALIZACIÓN COMPLETA...")
    
    # Descargar datasets
    check_for_updates()
    
    # Subir a GCP
    upload_to_gcp()
    
    # Re-entrenar
    result = retrain_model()
    
    print("\n✅ Actualización forzada completada")
    return result

if __name__ == "__main__":
    # Inicializar GCP si está disponible
    if GCP_AVAILABLE:
        gcp_utils.init_gcp()
    
    # Ejecutar ciclo de actualización
    result = asyncio.run(auto_update_cycle())
    print(f"\nResultado: {result['status']}")