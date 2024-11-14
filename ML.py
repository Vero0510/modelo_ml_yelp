# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor

# Inicializamos FastAPI
app = FastAPI()

# Cargamos los datos
df_yelp = pd.read_parquet("D:\\Desktop\\proyecto_grupal\\sprint1\\df_yelp.parquet")
df_atributos = pd.read_parquet("D:\\Desktop\\proyecto_grupal\\sprint1\\df_atributos.parquet")

# Procesamos los datos y entrenamos el modelo
city_counts = df_yelp['city'].value_counts()
top_cities = city_counts.nlargest(5).index
df_filtered = df_yelp[df_yelp['city'].isin(top_cities)]

encoder = OneHotEncoder(sparse_output=False)
city_encoded = encoder.fit_transform(df_filtered[['city']])
city_encoded_df = pd.DataFrame(city_encoded, columns=encoder.categories_[0])
df_filtered = pd.concat([df_filtered, city_encoded_df], axis=1)
df = pd.DataFrame(df_filtered)

df_filtered_selected = df_filtered[['name', 'city', 'review_count', 'funny']]
df_atributos_selected = df_atributos[["RestaurantsGoodForGroups","BusinessAcceptsCreditCards","GoodForKids","Smoking","BusinessParking"]]

preprocessor = ColumnTransformer(
    transformers=[
        ('city', OneHotEncoder(), ['city']),
        ('num', 'passthrough', ['review_count', 'funny', 'RestaurantsGoodForGroups',
                                'BusinessAcceptsCreditCards', 'GoodForKids', 'Smoking', 'BusinessParking'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

X = pd.concat([df_filtered_selected, df_atributos_selected], axis=1)
y = df_filtered['stars_y']
X = X.dropna(subset=["name", "city", "review_count", "funny"])
y = y[X.index]
model.fit(X, y)

# Definimos el modelo de entrada
class Caracteristicas(BaseModel):
    city: str
    review_count: int
    funny: int
    RestaurantsGoodForGroups: int
    BusinessAcceptsCreditCards: int
    GoodForKids: int
    Smoking: int
    BusinessParking: int

# Ruta para realizar la predicción
@app.post("/predict/")
def predecir_restaurante(caracteristicas: Caracteristicas):
    try:
        caracteristicas_dict = caracteristicas.dict()
        caracteristicas_df = pd.DataFrame([caracteristicas_dict])

        X_transformed = model.named_steps['preprocessor'].transform(caracteristicas_df)
        X_full_transformed = model.named_steps['preprocessor'].transform(X)
        cos_similarities = cosine_similarity(X_full_transformed, X_transformed)
        idx_mas_similar = np.argmax(cos_similarities)

        restaurante_similar = df.iloc[idx_mas_similar]
        return {
            "name": restaurante_similar['name'],
            "stars_y": restaurante_similar['stars_y']
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

