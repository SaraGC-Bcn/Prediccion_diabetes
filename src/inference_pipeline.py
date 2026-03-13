'''==== ARCHIVO PARA REALIZAR LA PREDICCIÓN CON EL MODELO ENTRENADO ====
Este archivo contiene el pipeline para realizar la predicción con el modelo entrenado
y los datos transofrmados. '''

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.feature_pipeline import transform_feature_pipeline
import joblib

#cargamos los artefactos necesarios para realizar la predicción
param = joblib.load('models/feature_pipeline.pkl')
model = joblib.load('models/randforest_model.pkl')

def predict_diabetes(X: pd.DataFrame) -> tuple[str, float]:
    '''Esta función, a partir de un DataFrame, lo transforma y aplica
    el modelo para realizar la predicción. Devuelve una tupla con la clase
    predicha y la probabilidad que el paciente tenga diabetes.'''

    #Aplicamos las transformaciones necesarias a los datos cargados
    X_transformada = transform_feature_pipeline (X, param)

    #Realizamos la predicción con el modelo entrenado
    prediction = model.predict(X_transformada)[0]
    probability = model.predict_proba(X_transformada)[0][1]

    label = 'Diabetes' if prediction == 1 else 'No Diabetes'

    return label, probability


