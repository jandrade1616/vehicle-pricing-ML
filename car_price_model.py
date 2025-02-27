#!/usr/bin/env python
# coding: utf-8

# ----

# ----

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Regresión Lineal
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import time

# Arbol de decisiones:
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import GridSearchCV

# Lightgbm
import lightgbm as lgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Graficos 
import plotly.graph_objects as go
import plotly.express as px
import plotly.express as px
import plotly.graph_objects as go


# In[2]:


# Cargar datos
file_path = '/datasets/car_data.csv'  # Cambia la ruta según tu archivo
car_data = pd.read_csv(file_path)
car_data.info()
car_data.head()
car_data.describe()


# In[3]:


# Eliminación de columnas irrelevantes
columns_to_drop = ['DateCrawled', 'DateCreated', 'LastSeen', 'NumberOfPictures', 'PostalCode']
car_data_cleaned = car_data.drop(columns=columns_to_drop)


# In[4]:


# Rango lógico para años de matriculación
car_data_cleaned = car_data_cleaned[
    (car_data_cleaned['RegistrationYear'] >= 1900) &
    (car_data_cleaned['RegistrationYear'] <= 2023)
]


# In[5]:


# Manejo de valores nulos
categorical_columns = ['VehicleType', 'Gearbox', 'FuelType', 'NotRepaired', 'Model']
car_data_cleaned[categorical_columns] = car_data_cleaned[categorical_columns].fillna('unknown')


# In[6]:


# Eliminación de los precios igual a 0
car_data_cleaned = car_data_cleaned[car_data_cleaned['Price'] > 0]


# In[7]:


# Separar características y variable objetivo
X = car_data_cleaned.drop(columns=['Price'])
y = car_data_cleaned['Price']


# In[8]:


# Codificación de variables categóricas
categorical_cols = X.select_dtypes(include='object').columns  # Seleccionar columnas categóricas
encoder = OneHotEncoder(drop=None, sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[categorical_cols])  # Codificar columnas categóricas

# Converción de los datos codificados a un DataFrame
X_encoded_df = pd.DataFrame(
    X_encoded,
    columns=encoder.get_feature_names(categorical_cols)  # Usa get_feature_names si get_feature_names_out no funciona
)

# Seleccionar columnas numéricas
X_numeric = X.select_dtypes(include=['int64', 'float64'])

# Combinar datos codificados y numéricos
X_prepared = pd.concat(
    [X_numeric.reset_index(drop=True), X_encoded_df.reset_index(drop=True)],
    axis=1
)

# Verificar dimensiones
print(f"X_prepared shape: {X_prepared.shape}")
print(f"y shape: {y.shape}")


# In[9]:


# Divición en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_prepared, y, test_size=0.2, random_state=42)

# Verificar las dimensiones
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")


# In[10]:


# Reducción del tamaño del conjunto de datos para pruebas
X_train_sample = X_train.sample(n=50000, random_state=42)
y_train_sample = y_train.reindex(X_train_sample.index)  # Alinear índices con reindex
X_test_sample = X_test.sample(n=10000, random_state=42)
y_test_sample = y_test.reindex(X_test_sample.index)


# In[11]:


from sklearn.impute import SimpleImputer

# Manejo de valores faltantes
num_imputer = SimpleImputer(strategy="median")
X_train_sample = num_imputer.fit_transform(X_train_sample)
X_test_sample = num_imputer.transform(X_test_sample)

y_train_sample = np.nan_to_num(y_train_sample, nan=np.nanmedian(y_train_sample))
y_test_sample = np.nan_to_num(y_test_sample, nan=np.nanmedian(y_test_sample))

# Verificar después de imputación
print("Valores NaN en X_train_sample:", np.sum(np.isnan(X_train_sample)))
print("Valores NaN en X_test_sample:", np.sum(np.isnan(X_test_sample)))
print("Valores NaN en y_train_sample:", np.sum(np.isnan(y_train_sample)))
print("Valores NaN en y_test_sample:", np.sum(np.isnan(y_test_sample)))


# In[12]:


# Entrenamiento del modelo de Regresión Lineal
start_time = time.time()
linear_model = LinearRegression()
linear_model.fit(X_train_sample, y_train_sample)
training_time_linear = time.time() - start_time

# Predicción y evaluación
y_pred_test_linear = linear_model.predict(X_test_sample)
rmse_test_linear = np.sqrt(mean_squared_error(y_test_sample, y_pred_test_linear))

# Resultados
print(f"Regresión Lineal - Tiempo de entrenamiento: {training_time_linear:.2f} segundos")
print(f"Regresión Lineal - RECM en conjunto de prueba: {rmse_test_linear:.2f}")


# In[13]:


# Entrenamiento del Árbol de Decisión
start_time = time.time()
tree_model = DecisionTreeRegressor(random_state=42, max_depth=10)  # Ajusta max_depth si es necesario
tree_model.fit(X_train_sample, y_train_sample)
training_time_tree = time.time() - start_time

# Predicción y evaluación
y_pred_test_tree = tree_model.predict(X_test_sample)
rmse_test_tree = np.sqrt(mean_squared_error(y_test_sample, y_pred_test_tree))

# Resultados
print(f"Árbol de Decisión - Tiempo de entrenamiento: {training_time_tree:.2f} segundos")
print(f"Árbol de Decisión - RECM en conjunto de prueba: {rmse_test_tree:.2f}")


# In[14]:


# Optimización de max_depth "Manual"
best_rmse = float('inf')
best_depth = None

for depth in range(5, 21):  # Rango de profundidades
    tree_model = DecisionTreeRegressor(random_state=42, max_depth=depth)
    tree_model.fit(X_train_sample, y_train_sample)
    y_pred_test_tree = tree_model.predict(X_test_sample)
    rmse_test_tree = np.sqrt(mean_squared_error(y_test_sample, y_pred_test_tree))
    
    if rmse_test_tree < best_rmse:
        best_rmse = rmse_test_tree
        best_depth = depth

print(f"Mejor profundidad: {best_depth}")
print(f"Mejor RECM: {best_rmse:.2f}")


# In[15]:


# Optimización de max_depth con "GridSearchCV "
# Definición de los parámetros para búsqueda
param_grid = {
    'max_depth': range(3, 11),  # Explora un rango ajustado
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Configuracion de GridSearchCV
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    scoring='neg_mean_squared_error',
    cv=3,  # Validación cruzada
    verbose=1
)

grid_search.fit(X_train_sample, y_train_sample)

# Obtención de los mejores parámetros
best_params = grid_search.best_params_
best_rmse = np.sqrt(-grid_search.best_score_)
print(f"Mejores parámetros: {best_params}")
print(f"Mejor RECM con GridSearchCV: {best_rmse:.2f}")


# In[16]:


start_time = time.time()

# Entrenamiento con los mejores parámetros
optimized_tree_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf']
)
optimized_tree_model.fit(X_train_sample, y_train_sample)

# Calcular el tiempo de entrenamiento
training_time = time.time() - start_time
print(f"LightGBM - Tiempo de entrenamiento: {training_time:.2f} segundos")

# Evaluación en conjunto de prueba
y_pred_test_optimized = optimized_tree_model.predict(X_test_sample)
rmse_test_optimized = np.sqrt(mean_squared_error(y_test_sample, y_pred_test_optimized))

print(f"RECM en conjunto de prueba con modelo optimizado: {rmse_test_optimized:.2f}")


# In[17]:


feature_names = X_train.columns  # Asegúrate de definir esto correctamente
X_train_sample = pd.DataFrame(X_train_sample, columns=feature_names)
X_test_sample = pd.DataFrame(X_test_sample, columns=feature_names)

# Paso 1: Identificar columnas categóricas
categorical_columns = X_train_sample.select_dtypes(include=['object', 'category']).columns

# Paso 2: Codificación con One-Hot Encoding
X_train_encoded = pd.get_dummies(X_train_sample, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test_sample, columns=categorical_columns, drop_first=True)

# Alinear columnas entre conjunto de entrenamiento y prueba
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1)
X_test_encoded.fillna(0, inplace=True)  # Completar valores faltantes con ceros

# Paso 3: Entrenar LightGBM
lgb_model = LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=7)

# Entrenamiento
lgb_model.fit(X_train_encoded, y_train_sample)

# Registrar el tiempo de inicio
start_time = time.time()

# Entrenamiento del modelo
lgb_model.fit(X_train_encoded, y_train_sample)

# Calcular el tiempo de entrenamiento
training_time = time.time() - start_time
print(f"LightGBM - Tiempo de entrenamiento: {training_time:.2f} segundos")

# Predicciones y evaluación
y_pred = lgb_model.predict(X_test_encoded)
rmse = np.sqrt(mean_squared_error(y_test_sample, y_pred))
print(f"LightGBM - RMSE en conjunto de prueba: {rmse:.2f}")


# In[18]:


from sklearn.ensemble import RandomForestRegressor

# Configurar y entrenar el modelo
rf_model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)

# Registrar el tiempo de inicio
start_time = time.time()

# Entrenamiento
rf_model.fit(X_train_encoded, y_train_sample)

# Calcular el tiempo de entrenamiento
training_time = time.time() - start_time
print(f"Bosque Aleatorio - Tiempo de entrenamiento: {training_time:.2f} segundos")

# Evaluación
y_pred_rf = rf_model.predict(X_test_encoded)
rmse_rf = np.sqrt(mean_squared_error(y_test_sample, y_pred_rf))
print(f"Bosque Aleatorio - RMSE: {rmse_rf:.2f}")


# In[19]:


from catboost import CatBoostRegressor
import time

# Configurar el modelo de CatBoost
catboost_model = CatBoostRegressor(iterations=100, depth=7, learning_rate=0.1, random_seed=42, verbose=0)

# Entrenamiento
start_time = time.time()
catboost_model.fit(X_train_sample, y_train_sample, cat_features=list(categorical_columns))
training_time_catboost = time.time() - start_time

# Predicciones
y_pred_catboost = catboost_model.predict(X_test_sample)

# Evaluación
rmse_catboost = np.sqrt(mean_squared_error(y_test_sample, y_pred_catboost))

# Resultados
print(f"CatBoost - Tiempo de entrenamiento: {training_time_catboost:.2f} segundos")
print(f"CatBoost - RMSE en conjunto de prueba: {rmse_catboost:.2f}")


# In[20]:


from xgboost import XGBRegressor

# Configurar el modelo de XGBoost
xgb_model = XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbosity=0)

# Entrenamiento
start_time = time.time()
xgb_model.fit(X_train_encoded, y_train_sample)
training_time_xgb = time.time() - start_time

# Predicciones
y_pred_xgb = xgb_model.predict(X_test_encoded)

# Evaluación
rmse_xgb = np.sqrt(mean_squared_error(y_test_sample, y_pred_xgb))

# Resultados
print(f"XGBoost - Tiempo de entrenamiento: {training_time_xgb:.2f} segundos")
print(f"XGBoost - RMSE en conjunto de prueba: {rmse_xgb:.2f}")


# In[21]:


# Resultados obtenidos previamente
rmse_test_linear = 2319.25
rmse_test_optimized = 2301.47
rmse = 2309.68
rmse_rf = 2300.99
rmse_catboost = 2303.00
rmse_xgb = 2319.24

training_time_linear = 1.48
training_time_tree = 0.14
training_time_lgbm = 2.7
training_time_rf = 32.25
training_time_catboost = 0.92
training_time_xgb = 72.75

# Crear un DataFrame para comparar resultados
model_comparison = pd.DataFrame({
    'Modelo': ['Regresión Lineal', 'Árbol de Decisión', 'LightGBM', 'Bosque Aleatorio', 'CatBoost', 'XGBoost'],
    'RMSE': [rmse_test_linear, rmse_test_optimized, rmse, rmse_rf, rmse_catboost, rmse_xgb],
    'Tiempo de Entrenamiento (s)': [training_time_linear, training_time_tree, training_time_lgbm, training_time_rf, training_time_catboost, training_time_xgb]
})

# Ordenar por RMSE ascendente
model_comparison = model_comparison.sort_values(by='RMSE', ascending=True)

# Mostrar tabla
print(model_comparison)


# In[22]:


pip install plotly


# In[23]:


# Crear tabla con colores personalizados
fig = go.Figure(data=[go.Table(
    header=dict(
        values=list(model_comparison.columns),
        fill_color='darkblue',
        font=dict(color='white'),
        align='left'
    ),
    cells=dict(
        values=[model_comparison[col] for col in model_comparison.columns],
        fill_color=['lightblue', 'lavender'],  # Alternar colores
        align='left'
    ))
])

fig.show()


# In[24]:


fig = px.bar(
    model_comparison,
    x='Modelo',
    y='RMSE',
    color='Modelo',
    title='Comparación de RMSE entre Modelos',
    text='RMSE'  # Agrega etiquetas para los valores de RMSE
)

# Mejorar el diseño
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

# Mostrar el gráfico
fig.show()


# In[25]:


# Crear la gráfica de barras para el tiempo de entrenamiento
fig = px.bar(
    model_comparison,
    x='Modelo',
    y='Tiempo de Entrenamiento (s)',
    title='Comparación de Tiempos de Entrenamiento entre Modelos',
    labels={'Tiempo de Entrenamiento (s)': 'Tiempo de Entrenamiento (s)', 'Modelo': 'Modelos'},
    color='Modelo',
    text='Tiempo de Entrenamiento (s)'  # Mostrar los valores directamente en las barras
)

# Personalizar el diseño
fig.update_layout(
    template='plotly_white',
    xaxis_title='Modelos',
    yaxis_title='Tiempo de Entrenamiento (s)'
)

# Mostrar la gráfica
fig.show()


# In[26]:


# Crear el gráfico de barras agrupadas
fig = go.Figure()

# Agregar las barras para RMSE
fig.add_trace(go.Bar(
    x=model_comparison['Modelo'],
    y=model_comparison['RMSE'],
    name='RMSE',
    marker_color='lightblue'
))

# Agregar las barras para el Tiempo de Entrenamiento
fig.add_trace(go.Bar(
    x=model_comparison['Modelo'],
    y=model_comparison['Tiempo de Entrenamiento (s)'],
    name='Tiempo de Entrenamiento (s)',
    marker_color='lavender'
))

# Personalizar el diseño
fig.update_layout(
    title='Comparación de RMSE y Tiempo de Entrenamiento entre Modelos',
    xaxis_title='Modelo',
    yaxis_title='Valores',
    barmode='group',  # Agrupar las barras
    template='plotly_white'
)

# Mostrar el gráfico
fig.show()

