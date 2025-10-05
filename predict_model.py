import sys
import pandas as pd
import joblib

# Cargar el modelo entrenado (imputador + clasificador) desde el archivo
model_bundle = joblib.load("exoplanet_model.pkl")
imputer = model_bundle["imputer"]
model = model_bundle["classifier"]

# Definir las columnas/features esperadas
features = ['period', 'duration', 'depth', 'radius', 'insolation', 'teff', 'srad']

if len(sys.argv) > 1:
    # Modo 1: Leer datos de un archivo CSV pasado como argumento
    input_path = sys.argv[1]
    new_data = pd.read_csv(input_path)
    # Comprobamos que contiene las columnas necesarias:
    for col in features:
        if col not in new_data.columns:
            raise ValueError(f"Missing column '{col}' in input data")
    # Aplicar la misma imputación de valores faltantes que en entrenamiento
    new_data_imputed = pd.DataFrame(imputer.transform(new_data[features]), columns=features)
    # Realizar las predicciones
    preds = model.predict(new_data_imputed.values)
    # Añadir las predicciones al dataframe y guardar/mostrar resultados
    new_data['predicted_label'] = preds
    print("Predictions for new data:")
    print(new_data)
    # Opcional: guardar a CSV
    output_file = "predictions_output.csv"
    new_data.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
else:
    # Modo 2: Sin archivo, usar un ejemplo manual
    example = pd.DataFrame([{
        'period': 10.0,      # periodo orbital en días
        'duration': 5.0,     # duración del tránsito en horas
        'depth': 500.0,      # profundidad del tránsito en ppm
        'radius': 2.0,       # radio planetario en radios de la Tierra
        'insolation': 150.0, # insolación en unidades de la Tierra
        'teff': 5700.0,      # temperatura efectiva de la estrella (K)
        'srad': 1.0          # radio de la estrella en radios solares
    }])
    # Imputar valores faltantes si los hubiera (en este ejemplo no hay NaN)
    example_imputed = pd.DataFrame(imputer.transform(example[features]), columns=features)
    pred = model.predict(example_imputed.values)[0]
    print("Example input:\n", example)
    print("Predicted label for the example input:", pred)