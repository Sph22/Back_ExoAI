from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, ValidationError
import predict_model  # Importa el módulo con la lógica de predicción (predict_model.py)

app = FastAPI()
#hola
# Modelo de datos de entrada utilizando Pydantic
class InputData(BaseModel):
    period: float
    duration: float
    depth: float
    radius: float
    insolation: float
    teff: float
    srad: float

@app.post("/predict")
async def predict(data_request: Request):
    # Leer el cuerpo JSON crudo de la petición
    data = await data_request.json()
    try:
        # Si es una lista de objetos, validarlos individualmente
        if isinstance(data, list):
            inputs = [InputData(**item) for item in data]
        # Si es un solo objeto (dict), envolverlo en una lista de tamaño 1
        elif isinstance(data, dict):
            inputs = [InputData(**data)]
        else:
            # Formato no esperado
            raise HTTPException(status_code=400, detail="Formato de entrada no válido. Se esperaba un objeto JSON o una lista de objetos.")
    except ValidationError as e:
        # Capturar errores de validación de Pydantic y devolver como 422
        raise HTTPException(status_code=422, detail=e.errors())
    
    # Para cada entrada validada, obtener la predicción del modelo
    predictions = []
    for input_obj in inputs:
        # Cargar el modelo en cada solicitud para usar la versión más reciente
        # (Por ejemplo, si hubiera un archivo de modelo: model = joblib.load('modelo.pkl'))
        # Ejecutar la predicción utilizando la lógica de predict_model.py
        result = predict_model.predict(input_obj)  
        predictions.append(result)
    
    # Formar la respuesta dependiendo de si era entrada única o múltiple
    if len(predictions) == 1:
        return {"prediction": predictions[0]}
    else:
        return {"predictions": predictions}

# Punto de entrada para ejecutar con Uvicorn localmente (por ejemplo, uvicorn main:app --reload)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)