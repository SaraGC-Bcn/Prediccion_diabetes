'''==== ARCHIVO PARA PROBAR SI EL PIPELINE DE INFERENCE FUNCIONA ====
Con datos que introducimos como ejemplo manualmente, probamos que el pipeline
de inference funcione correctamnete.'''

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.inference_pipeline import predict_diabetes

data = pd.DataFrame({
    "Pregnancies": [4],
    "Glucose": [140],
    "BloodPressure": [78],
    "SkinThickness": [15],
    "Insulin": [68],
    "BMI": [29.0],
    "DiabetesPedigreeFunction": [0.45],
    "Age": [25]
})

label, probability = predict_diabetes(data)

print('\nEl paciente tiene: ', label)
print(f'Probabilidad de tener diabetes: {probability:.1%}')
print('')