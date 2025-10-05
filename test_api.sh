#!/bin/bash

# URL base de tu API en Render
BASE_URL="https://back-exoai.onrender.com"

echo "=========================================="
echo "ExoAI API - Testing Optimizado"
echo "=========================================="
echo ""

# 1. Health check
echo "1. Health Check"
curl -s "${BASE_URL}/health" | jq '.'
echo ""
echo ""

# 2. Obtener primeros 3 registros con nombres
echo "2. Obtener primeros 3 registros con nombres (NUEVO ENDPOINT)"
curl -s "${BASE_URL}/datasets/first?n=3" | jq '.'
echo ""
echo ""

# 3. Preview de datasets
echo "3. Preview de datasets (2 filas)"
curl -s "${BASE_URL}/datasets/preview?n=2" | jq '.datasets.cumulative.head'
echo ""
echo ""

# 4. Test de predicción
echo "4. Test de predicción"
curl -s "${BASE_URL}/predict/test" | jq '.'
echo ""
echo ""

# 5. Predicción simple
echo "5. Predicción simple"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "period": 3.52,
    "duration": 2.5,
    "depth": 0.0013,
    "radius": 1.2,
    "insolation": 1.05,
    "teff": 5778,
    "srad": 1.0
  }' | jq '.'
echo ""
echo ""

# 6. Status de entrenamiento
echo "6. Status de entrenamiento"
curl -s "${BASE_URL}/train/status" | jq '.'
echo ""
echo ""

# 7. Listar versiones de modelos en GCP (si está configurado)
echo "7. Versiones de modelos en GCP"
curl -s "${BASE_URL}/cloud/models/versions" | jq '.'
echo ""
echo ""

echo "=========================================="
echo "Testing completado"
echo "=========================================="