import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# !!Bienvenidos a nuestro predictor de Netflix!!!

"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Datos ficticios de películas y preferencias
datos = {
    'Género': ['Acción', 'Comedia', 'Drama', 'Acción', 'Comedia', 'Drama', 'Acción', 'Comedia'],
    'Director': ['Director1', 'Director2', 'Director3', 'Director1', 'Director2', 'Director3', 'Director1', 'Director2'],
    'Año': [2010, 2015, 2018, 2012, 2017, 2019, 2014, 2016],
    'Duración': [120, 95, 110, 130, 100, 105, 115, 125],
    'Gusto': ['Sí', 'No', 'Sí', 'Sí', 'No', 'Sí', 'No', 'Sí']
}

# Convertir los datos a un DataFrame
df = pd.DataFrame(datos)

# Codificar las variables categóricas
df_encoded = pd.get_dummies(df, columns=['Género', 'Director'])

# Dividir los datos en características (X) y variable objetivo (y)
X = df_encoded.drop('Gusto', axis=1)
y = df_encoded['Gusto']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de Random Forest
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo_rf.predict(X_test)

# Calcular la precisión del modelo
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precision)

