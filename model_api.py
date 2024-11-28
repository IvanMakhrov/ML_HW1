from fastapi import FastAPI, UploadFile, File, responses
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import uvicorn
from io import StringIO
from sklearn.base import BaseEstimator
from datetime import datetime
import numpy as np


app = FastAPI()

class CleanData(BaseEstimator):
    def __init__(self):
        pass


    @staticmethod
    def cat_to_float(x):
        if isinstance(x, float) or isinstance(x, int):
            return float(x)

        x = x.split()

        if len(x) == 2:
            return float(x[0])
        elif len(x) == 1 and isinstance(x[0], str) and isinstance(x[0], str):
            return np.nan


    @staticmethod
    def torque_to_float(x):
        if isinstance(x, float):
            return np.nan

        x = x.lower()
        for a, b in zip(['~', '@', '(', ')', 'rpm', 'at', ',', '/'], ['-', '', '', '', '', '', '', '']):
            x = x.replace(a, b)

        x = x.strip().split()

        if len(x) == 1:
            x[0] = float(x[0].replace('nm', ''))
            x.append(np.nan)
        elif len(x) == 2:
            if 'nm' in x[0] and 'kgm' in x[0]:
                x[0] = float(max(map(float, x[0].replace('nm', ' ').replace('kgm', ' ').strip().split())))
            elif 'nm' in x[0] and isinstance(x[1], float):
                x[0] = float(x[0].replace('nm', ''))
            elif '+' in x[1]:
                x[1] = float(x[1].split('+')[0])
            elif 'nm' in x[0]:
                x[0] = float(x[0].replace('nm', ''))
            elif 'kgm' in x[0] or 'kgm' in x[1]:
                x[0] = round(float(x[0].replace('kgm', '')) * 9.80665, 2)
                x[1] = x[1].replace('kgm', '')

            if isinstance(x[1], float):
                pass
            elif len(x[1].split('-')) == 2:
                x[1] = sum(map(float, x[1].split('-')))/2
            elif len(x[1].split('-')) == 1:
                x[1] = float(x[1])

        elif len(x) == 3:
            if x[1] == 'nm':
                x.remove('nm')
                x[0] = float(x[0])
            elif x[1] == 'kgm':
                x.remove('kgm')
                x[0] = float(x[0]) * 9.80665

            if '-' in x[1]:
                x[1] = sum(map(float, x[1].split('-')))/2

        if isinstance(x[0], str):
            x[0] = float(x[0].replace('nm', ''))

        return f'{round(float(x[0]), 2)} {round(float(x[1]), 2)}'


    @staticmethod
    def define_age(age):
        if age <= 3:
            return 'New'
        elif age <= 9:
            return 'Used'
        else:
            return 'Old'

    @staticmethod
    def define_performance(max_power):
        if max_power >= 200:
            return 'High Performance'
        elif max_power >= 100:
            return 'Medium Performance'
        else:
            return 'Low Performance'


    def torque_processing(self, X):
        X['torque'] = X['torque'].apply(lambda x: self.torque_to_float(x))
        X['max_torque_rpm'] = X['torque'].astype(str).str.split(' ').str[1].astype(float)
        X['torque'] = X['torque'].astype(str).str.split(' ').str[0].astype(float)

        return X


    def object_to_float_processing(self, X):
        for col in ['mileage', 'engine', 'max_power']:
            X[col] = X[col].apply(lambda x: self.cat_to_float(x))

        return X



    def process_name(self, X):
        brand_model = pd.DataFrame([(car.split()[0], car.split()[1]) for car in X['name']], columns = ['brand', 'model'])

        X['brand'] = brand_model['brand']
        X['model'] = brand_model['model']

        X = X.drop(['name'], axis=1)

        return X


    def generate_columns(self, X):
        current_year = datetime.now().year
        X['age'] = current_year - X['year']
        X['km_per_year'] = X['km_driven'] / X['age']
        X['power_per_litre'] = X['max_power']/ X['engine']/1000
        X['year'] = X['year']**2
        X['age_category'] = X['age'].apply(self.define_age)
        X['performance_category'] = X['max_power'].apply(self.define_performance)

        return X


    def fit(self, X, Y):
        return self


    def transform(self, X):
        X = self.torque_processing(X)
        X = self.object_to_float_processing(X)
        X = self.process_name(X)
        X = self.generate_columns(X)

        X = pd.DataFrame(X)

        return X


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    car_info = pd.DataFrame([dict(item).values()],columns=list(dict(item).keys()))

    return model.predict(car_info)


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    contents = await file.read()

    df = pd.read_csv(StringIO(contents.decode("utf-8")))
    df_predict = df.copy(deep=True)
    
    selling_price_forecast = model.predict(df_predict)
    df['selling_price_forecast'] = selling_price_forecast

    df.to_csv('service_files/test_forecast.csv', index=False) 

    return responses.FileResponse('service_files/test_forecast.csv', media_type='text/csv', filename='test_forecast.csv')

if __name__ == "__main__":
    model = joblib.load('pipeline.pickle')
    uvicorn.run(app, host="127.0.0.1", port=8000)
