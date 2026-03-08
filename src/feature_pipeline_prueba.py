'''=========ARCHIVO PARA COMPROBACION DE FUNCIONES=========
Este archivo permite probar las funciones que se han creado en feature_pipeline.py'''

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.feature_pipeline import fit_feature_pipeline, transform_feature_pipeline


diabetes=pd.read_csv('data/processed/diabetes_postEDA.csv')
X=diabetes.drop(columns=['Outcome'])
y=diabetes['Outcome']
X_train, X_test, y_train, y_test = train_test_split (X, y,test_size = 0.2, random_state=42,
                                                     stratify=y)

num_cols = X_train.select_dtypes(include='number').columns.tolist()
winsor_columns = ['BloodPressure', 'SkinThickness', 'Insulin','DiabetesPedigreeFunction']

param = fit_feature_pipeline(X_train, num_cols, winsor_columns)
X_test_trans = transform_feature_pipeline(X_test, param)

print(X_test_trans.head())