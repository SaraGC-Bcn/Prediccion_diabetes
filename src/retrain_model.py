'''==== ARCHIVO PARA REENTRENAR EL MODELO CON TODOS LOS DATOS DE ORIGEN ====
Este archivo contiene todas las instrucciones para reentrenar el modelo
con todos los datos de origen. '''

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.feature_pipeline import cleaning_training_data, fit_feature_pipeline, transform_feature_pipeline
from src.training_pipeline import train_model_randforest
from imblearn.over_sampling import SMOTE
import joblib


#cargar los datos de origen
data_raw = pd.read_csv('data/raw/diabetes_original.csv')

#realizamos limpieza de líneas duplicadas y aquellas con 3 o más valores 0 en las columnas que no pueden tomarlo
data_clean = cleaning_training_data(data_raw)

#separamos X e y
X = data_clean.drop(columns='Outcome')
y = data_clean['Outcome']

#separamos en tipos de columnas, en este caso, solo numéricas
#y columnas para la winsorización
num_cols = X.select_dtypes(include='number').columns.tolist()
winsor_cols = ['BloodPressure', 'SkinThickness', 'Insulin','DiabetesPedigreeFunction']

#aplicamos las transformaciones necesarias a los datos cargados. Aplicamos fit, transform
# y guardamos param como feature_pipeline.pkl. 
param = fit_feature_pipeline(X, num_cols,winsor_cols)
X_transformada = transform_feature_pipeline (X, param)
joblib.dump(param, 'models/feature_pipeline.pkl')

#aplicamos SMOTE para balancear el dataset
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_transformada, y)

#reentrenamos el modelo con los datos transformados y balanceados con SMOTE, 
# y guardamos el modelo reentrenado como randforest_model.pkl
modelo_reentrenado = train_model_randforest (X_smote, y_smote)
joblib.dump(modelo_reentrenado, 'models/randforest_model.pkl')

print('\n✅ Modelo reentrenado con éxito!')
print('✅ Artefactos guardados en la carpeta models')
