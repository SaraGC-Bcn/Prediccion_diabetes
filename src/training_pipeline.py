'''==== ARCHIVO PARA ENTRENAR AL MODELO CON LOS MEJORES HIPERPARAMETROS QUE VIMOS EN EL COLAB ====
Este archivo contiene el modelo con los hiperparámetros seleccionados como mejores y está listo 
para ser entrenado con el X_train que se le pase. '''

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model_randforest (X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        max_features="sqrt", 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/randforest_model.pkl')

    return model