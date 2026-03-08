import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def apply_domain_rules(X):
    '''Función para aplicar reglas de dominio a todas las variables, reemplazando los valores que 
    no cumplen con las reglas por NaN'''
    X = X.copy()
    #Pregnancies
    X.loc[(X['Pregnancies']<0) | (X['Pregnancies']>30), 'Pregnancies']=np.nan
    #Glucose
    X.loc[(X['Glucose']<=0), 'Glucose']=np.nan
    #BloodPressure
    X.loc[(X['BloodPressure']<=0), 'BloodPressure']=np.nan
    #SkinThickness
    X.loc[(X['SkinThickness']<=0) | (X['SkinThickness']>110), 'SkinThickness']=np.nan
    #Insulin
    X.loc[(X['Insulin']<=0), 'Insulin']=np.nan
    #BMI
    X.loc[(X['BMI']<=0) | (X['BMI']>80), 'BMI']=np.nan
    #DiabetesPedigreeFunction
    X.loc[(X['DiabetesPedigreeFunction']<0), 'DiabetesPedigreeFunction']=np.nan
    #Age
    X.loc[(X['Age']<=0) | (X['Age']>120), 'Age']=np.nan
    return X

def fit_imputer_nan_to_mediana (X, columns_list):
    '''Función para entrenar los imputers de NaN a mediana de todas las variables numericas'''
    imp = SimpleImputer(strategy='median')
    imp.fit(X[columns_list])
    return imp

def transform_imputer_nan_to_mediana (X, imp, columns_list):
    '''Aplicamos los imputers entrenados a un nuevo X '''
    #transformamos los valores NaN a la mediana de cada columna
    X = X.copy()
    X[columns_list] = imp.transform(X[columns_list])
    return X

def limit_outliers (X, columns_list):
    '''Funcion para calcular los limites en la winsorización con X_train'''
    limit_summary={}
    for col in columns_list:
        limit_summary[col] = (X[col].quantile(0.01), X[col].quantile(0.99))
    return limit_summary

def winsor (X, limit_summary):
    '''Función para hacer la winsorización de todas las columnas numéricas con los limites entrenados 
    con X_train'''
    X=X.copy()
    for col, (low, high) in limit_summary.items():
        #.clip asigna a valores <bottom_limit, = bottom_limit y para valores >top_limit, = top_limit
        X[col]=X[col].clip(lower=low, upper=high)
    return X

def fit_rscaler(X,columns_list):
    '''Función para entrenar al escalador con X_train'''
    rscaler = RobustScaler()
    rscaler.fit(X[columns_list])
    return rscaler

def transform_rscaler(X, rscaler, columns_list):
    '''Función para aplicar la transformación entrenada con X_train'''
    X=X.copy()
    X[columns_list] = rscaler.transform(X[columns_list])
    return X



def fit_feature_pipeline(X_train, num_cols,winsor_cols):
    '''Aprende los parametros de los transformadores con X_train.
    Devuelve un diccionario con: las columnas que trabajamos, imputer,
    limites para la winsorizacion y RobustScaler. '''
    #1)Reglas de dominio. Lo que es imposible, pasa a NaN
    X = apply_domain_rules(X_train)

    #2)Entrenamos el objeto SimpleImputer con X_train y lo aplicamos a X_train
    imputer = fit_imputer_nan_to_mediana (X, num_cols)
    X = transform_imputer_nan_to_mediana (X, imputer, num_cols)

    #3)Calculamos el diccionario con los limites para la winsorizacion con X_train y lo aplicamos a X_train
    winsor_limits = limit_outliers (X, winsor_cols)
    X = winsor (X, winsor_limits)

    #4)Entrenamos el objeto RScaler con X_train (no hace falta transformar a X_train pq es lo ultimo)
    rscaler = fit_rscaler(X,num_cols)

    param = {
        'num_cols':num_cols,
        'winsor_cols':winsor_cols,
        'imputer':imputer,
        'winsor_limits':winsor_limits,
        'rscaler':rscaler
    }
    return param


def transform_feature_pipeline (X, param):
     '''Aplica las transformaciones a cualquier juego de datos (X_test o datos nuevos)
     con los parámetros que se han entrenado con X_train obtenidos en la función
     fit_feature_pipeline.
     Devuelve los datos transformados'''

     #1)Detección de columnas faltantes (si falta alguna, se para el proceso)
     num_cols = param['num_cols']
     missing_cols = [col for col in num_cols if col not in X.columns]
     if missing_cols:
        raise ValueError (f'Columnas faltantes necesarias para el modelo: {missing_cols}')

     #2)Seleción de columnas necesarias para la transformación y ordenación de las mismas como en entreno
     X = X[num_cols]

     #3)Reglas de dominio. Lo que es imposible, pasa a NaN
     X_transformada = apply_domain_rules(X)

     #4)Aplicamos SimpleImputer para pasar de NaN a la mediana calculada con X_train
     imputer = param['imputer']
     X_transformada = transform_imputer_nan_to_mediana (X_transformada, imputer, num_cols)

     #5)Aplicamos winsorización con los límites calculados con X_train
     winsor_limits = param['winsor_limits']
     X_transformada = winsor(X_transformada, winsor_limits)

     #6)Escalamos los datos con el Robust Scaler entrenado con X_train
     rscaler = param['rscaler']
     X_transformada = transform_rscaler(X_transformada, rscaler, num_cols)

     return X_transformada