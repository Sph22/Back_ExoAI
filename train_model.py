import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Cargar los datasets de Kepler, K2 y TESS desde archivos CSV
koi_file = "cumulative_2025.10.02_09.40.37.csv"   # Kepler KOI cumulative dataset
k2_file = "k2pandc_2025.10.02_09.42.15.csv"       # K2 confirmed & candidate dataset
toi_file = "TOI_2025.10.02_09.42.06.csv"          # TESS Objects of Interest dataset

print("Loading datasets...")
# Usamos comment='#' para ignorar las filas de comentario del archivo (cabeceras descriptivas de NASA)
koi_df = pd.read_csv(koi_file, comment='#')
k2_df = pd.read_csv(k2_file, comment='#')
toi_df = pd.read_csv(toi_file, comment='#')
print(f"Kepler KOI entries: {len(koi_df)}, K2 entries: {len(k2_df)}, TESS entries: {len(toi_df)}")

# 2. Filtrar K2 para tomar sólo la solución por defecto de cada objeto (default_flag == 1)
if 'default_flag' in k2_df.columns:
    k2_df = k2_df[k2_df['default_flag'] == 1].copy()
    print(f"K2 entries after default_flag filter: {len(k2_df)}")

# 3. Mapear las disposiciones/etiquetas de cada dataset a un esquema unificado (CONFIRMED, CANDIDATE, FALSE POSITIVE)
def map_koi_label(disposition):
    # Kepler dispositions are already 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'
    return disposition  # no change needed (we'll ensure consistency in capitalization though)
    
def map_k2_label(disposition):
    # K2 dispositions include 'CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE', and 'REFUTED'
    if disposition == 'REFUTED' or disposition == 'FALSE POSITIVE':
        return 'FALSE POSITIVE'
    elif disposition == 'CONFIRMED':
        return 'CONFIRMED'
    elif disposition == 'CANDIDATE':
        return 'CANDIDATE'
    else:
        return disposition  # unexpected cases
    
def map_toi_label(tfopwg_disp):
    # TESS TFOPWG dispositions: CP, KP = confirmed, PC, APC = candidate, FP, FA = false positive
    if tfopwg_disp in ['CP', 'KP']:
        return 'CONFIRMED'
    elif tfopwg_disp in ['PC', 'APC']:
        return 'CANDIDATE'
    elif tfopwg_disp in ['FP', 'FA']:
        return 'FALSE POSITIVE'
    else:
        return tfopwg_disp  # if any other appears
    
koi_df['Label'] = koi_df['koi_disposition'].apply(map_koi_label)
k2_df['Label'] = k2_df['disposition'].apply(map_k2_label)
toi_df['Label'] = toi_df['tfopwg_disp'].apply(map_toi_label)

# 4. Seleccionar y renombrar columnas de interés en cada dataframe
# Definimos las columnas deseadas en el conjunto final
# Nota: K2 no tiene explicitamente duration ni depth en su tabla, colocaremos NaN luego
koi_selected = pd.DataFrame({
    'period': koi_df['koi_period'],
    'duration': koi_df['koi_duration'],
    'depth': koi_df['koi_depth'],
    'radius': koi_df['koi_prad'],
    'insolation': koi_df['koi_insol'],
    'teff': koi_df['koi_steff'],
    'srad': koi_df['koi_srad'],
    'label': koi_df['Label']
})
k2_selected = pd.DataFrame({
    'period': k2_df['pl_orbper'],
    'duration': np.nan,                # no direct transit duration column in K2 dataset
    'depth': np.nan,                   # no direct transit depth column in K2 dataset
    'radius': k2_df['pl_rade'],
    'insolation': k2_df.get('pl_insol', np.nan),  # use .get in case column missing
    'teff': k2_df['st_teff'],
    'srad': k2_df['st_rad'],
    'label': k2_df['Label']
})
toi_selected = pd.DataFrame({
    'period': toi_df['pl_orbper'],
    'duration': toi_df['pl_trandurh'],
    'depth': toi_df['pl_trandep'],
    'radius': toi_df['pl_rade'],
    'insolation': toi_df['pl_insol'],
    'teff': toi_df['st_teff'],
    'srad': toi_df['st_rad'],
    'label': toi_df['Label']
})

# Unimos las tres fuentes de datos
data = pd.concat([koi_selected, k2_selected, toi_selected], ignore_index=True)
print(f"Total combined entries: {len(data)}")

# 5. Manejar valores faltantes en las características numéricas usando la mediana de cada columna
features = ['period', 'duration', 'depth', 'radius', 'insolation', 'teff', 'srad']
# Usamos SimpleImputer para rellenar NaN; entrenamos el imputer en todo el conjunto combinado
imputer = SimpleImputer(strategy='median')
data[features] = imputer.fit_transform(data[features])

# 6. Separar entrenamiento y prueba
X = data[features].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 7. Definir y entrenar el modelo de Random Forest
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

# 8. Evaluar el modelo en el conjunto de prueba
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test data: {accuracy:.4f}")
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))

# 9. Guardar el modelo entrenado y el imputador para futuras predicciones
# Guardamos ambos juntos en un pipeline manual: necesitamos aplicar el imputer antes del modelo al predecir nuevos datos.
# Una forma sencilla: empaquetar en un objeto (por ejemplo, tuple) o crear un Pipeline real de scikit-learn.
# Aquí empaquetamos en un tuple para simplicidad.
model_bundle = {
    "imputer": imputer,
    "classifier": model
}
joblib.dump(model_bundle, "exoplanet_model.pkl")
print("Trained model saved to exoplanet_model.pkl")

# 10. Ejemplo de uso del modelo entrenado con un nuevo dato (simulado)
# Vamos a tomar un ejemplo del propio conjunto de prueba para demostrar.
example_features = X_test[0]
true_label = y_test[0]
pred_label = model.predict([example_features])[0]
print("Example candidate features:", example_features)
print(f"True label: {true_label}  -> Model predicted: {pred_label}")