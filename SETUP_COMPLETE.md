# Configuración Completa - ExoAI con Google Cloud

## Resumen de Cambios

### Archivos Nuevos:
1. **`auto_update.py`** - Sistema de actualización automática
2. **`.github/workflows/auto-update.yml`** - GitHub Actions para cron job
3. **`space_launches.py`** - Actualizado con mejor manejo de rate limits

### Archivos Modificados:
- **`main.py`** - Agregados endpoints `/auto-update` y `/auto-update/force`
- **`space_launches.py`** - Cache aumentado a 60 min, rate limiting mejorado

---

## Problema del Service Account Key

Tu organización tiene bloqueada la creación de keys. Necesitas crear el key manualmente:

### Opción 1: Desde la Consola Web (RECOMENDADO)

1. Ve a: https://console.cloud.google.com/iam-admin/serviceaccounts?project=metal-sorter-474209-c9
2. Click en `exoai-service@metal-sorter-474209-c9.iam.gserviceaccount.com`
3. Pestaña "KEYS" → "ADD KEY" → "Create new key"
4. Selecciona JSON → CREATE
5. Descarga el archivo

### Opción 2: Desde Cloud Shell

Si tienes permisos de super admin:

```bash
# Levantar restricción temporalmente
gcloud resource-manager org-policies delete \
  iam.disableServiceAccountKeyCreation \
  --project=metal-sorter-474209-c9

# Crear key
gcloud iam service-accounts keys create gcp-key.json \
  --iam-account=exoai-service@metal-sorter-474209-c9.iam.gserviceaccount.com

# Descargar (usa el botón de download en Cloud Shell)
```

### Opción 3: Workload Identity Federation (Avanzado)

Si no puedes crear keys, contacta al admin de tu organización o usa Workload Identity Federation (más complejo).

---

## Configurar Render

### Variables de Entorno:

```bash
DATA_DIR=data
MODEL_PATH=exoplanet_model.pkl
TRAIN_TOKEN=algo_super_seguro
GCP_PROJECT_ID=metal-sorter-474209-c9
GCP_BUCKET_NAME=exoai-models-metal-sorter-474209-c9
```

### Secret File:

1. En Render Dashboard > Environment
2. Click "Add Secret File"
3. Configuración:
   - **Filename**: `/etc/secrets/gcp-key.json`
   - **Contents**: (pega TODO el contenido del JSON descargado)

---

## Configurar GitHub Actions

### 1. Agregar Secret en GitHub:

1. Ve a tu repo: https://github.com/Sph22/Back_ExoAI
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `TRAIN_TOKEN`
5. Value: `algo_super_seguro` (el mismo que en Render)

### 2. Crear el archivo de workflow:

```bash
mkdir -p .github/workflows
# Copia el contenido de auto-update.yml que te proporcioné
```

### 3. Activar GitHub Actions:

1. Ve a la pestaña "Actions" de tu repo
2. Si está desactivado, click "I understand my workflows, go ahead and enable them"

---

## Inicializar Firestore

Si aún no lo hiciste:

```bash
gcloud firestore databases create --location=us-central1 --project=metal-sorter-474209-c9
```

---

## Pruebas

### 1. Verificar que GCP esté funcionando:

```bash
curl https://back-exoai.onrender.com/health

# Respuesta esperada:
{
  "status": "healthy",
  "model_ready": true,
  "model_in_memory": true,
  "data_available": true,
  "gcp_status": "enabled"
}
```

### 2. Probar actualización automática (MANUAL):

```bash
curl -X POST "https://back-exoai.onrender.com/auto-update?token=algo_super_seguro"
```

### 3. Forzar actualización completa:

```bash
curl -X POST "https://back-exoai.onrender.com/auto-update/force?token=algo_super_seguro"
```

### 4. Hacer predicción (usa POST, no GET):

```bash
curl -X POST https://back-exoai.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "period": 3.52,
    "duration": 2.8,
    "depth": 4500,
    "radius": 2.5,
    "insolation": 150,
    "teff": 5800,
    "srad": 1.1
  }'

# Respuesta esperada:
{
  "prediction": {
    "prediction": "CANDIDATE",
    "confidence_percentage": 87.45,
    "probabilities": {
      "CANDIDATE": 87.45,
      "CONFIRMED": 8.32,
      "FALSE POSITIVE": 4.23
    }
  }
}
```

### 5. Entrenar modelo (usa POST, no GET):

```bash
curl -X POST "https://back-exoai.onrender.com/train?token=algo_super_seguro"
```

### 6. Probar Space Launches (con caché):

```bash
# Primera llamada (descarga desde API)
curl "https://back-exoai.onrender.com/launches/upcoming?limit=5"

# Segunda llamada (usa caché, más rápido)
curl "https://back-exoai.onrender.com/launches/upcoming?limit=5"
```

---

## Cómo Funciona la Automatización

### Flujo Automático (cada 6 horas):

1. **GitHub Actions** ejecuta el cron job
2. Hace POST a `/auto-update` con el token
3. El endpoint:
   - Descarga datasets desde NASA
   - Compara hashes (detecta cambios)
   - Si hay cambios:
     - Sube datasets a GCP
     - Re-entrena el modelo
     - Sube modelo nuevo a GCP
     - Guarda métricas en Firestore
   - Si NO hay cambios:
     - No hace nada, ahorra recursos

### Ventajas:

- **Datos siempre actualizados** sin intervención manual
- **Ahorro de costos**: Solo entrena si hay cambios reales
- **Cache inteligente**: Usa GCP para evitar descargas innecesarias
- **Histórico**: Firestore guarda todas las métricas de entrenamientos

### Verificar ejecuciones:

1. Ve a: https://github.com/Sph22/Back_ExoAI/actions
2. Verás el historial de ejecuciones
3. Click en cualquiera para ver logs detallados

---

## Solución de Problemas

### Error: "Method Not Allowed" en /predict o /train

**Causa**: Estás usando GET en vez de POST

**Solución**: Usa `-X POST` en curl:
```bash
curl -X POST https://back-exoai.onrender.com/predict -H "Content-Type: application/json" -d '{ ... }'
```

### Error: 429 en Space Launches

**Causa**: Rate limit de la API

**Solución**: Espera 60 minutos o usa el caché:
```bash
# El caché se usa automáticamente
curl "https://back-exoai.onrender.com/launches/upcoming?limit=10"
```

### Error: "GCP credentials no encontradas"

**Causa**: El Secret File no está configurado correctamente

**Solución**:
1. Verifica que el filename sea exactamente: `/etc/secrets/gcp-key.json`
2. Verifica que el contenido sea el JSON completo (no solo una parte)
3. Redeploy en Render

### Error: "Token inválido"

**Causa**: El token en Render y GitHub no coinciden

**Solución**:
1. Verifica `TRAIN_TOKEN` en Render
2. Verifica secret `TRAIN_TOKEN` en GitHub
3. Deben ser idénticos

---

## Mantenimiento

### Limpiar caché manualmente:

```bash
curl -X POST "https://back-exoai.onrender.com/cache/clear?token=algo_super_seguro"
```

### Ver versiones de modelos en GCP:

```bash
curl "https://back-exoai.onrender.com/cloud/models/versions"
```

### Descargar modelo específico desde GCP:

```bash
curl -X POST "https://back-exoai.onrender.com/cloud/models/download?version=20251005_123456&token=algo_super_seguro"
```

### Ver historial de entrenamientos:

```bash
curl "https://back-exoai.onrender.com/train/history?limit=10"
```

---

## Próximos Pasos

1. ✅ Crear Service Account Key manualmente
2. ✅ Configurar Secret File en Render
3. ✅ Configurar GitHub Actions con el secret TRAIN_TOKEN
4. ✅ Hacer commit y push de los archivos nuevos
5. ✅ Verificar que `/health` muestre `gcp_status: "enabled"`
6. ✅ Ejecutar `/auto-update/force` para primera sincronización
7. ✅ Esperar 6 horas y verificar que el cron job funcione

---

## Monitoreo

### Endpoints de Monitoreo:

```bash
# Estado general
curl https://back-exoai.onrender.com/health

# Estado de entrenamiento
curl https://back-exoai.onrender.com/train/status

# Estadísticas de lanzamientos
curl https://back-exoai.onrender.com/launches/stats

# Listar archivos en GCP
curl https://back-exoai.onrender.com/cloud/storage/list

# Ver versiones de modelos
curl https://back-exoai.onrender.com/cloud/models/versions
```

---

## Costos Estimados de GCP

- **Storage (1GB)**: ~$0.02/mes
- **Firestore (10k docs)**: ~$0.01/mes
- **Operaciones API**: ~$0.40/mes

**Total**: ~$1-5/mes con uso moderado

---

## Soporte

Si tienes problemas:

1. Revisa los logs de Render
2. Revisa los logs de GitHub Actions
3. Verifica las variables de entorno
4. Asegúrate de usar POST donde corresponde
5. Verifica que el archivo JSON de GCP esté completo