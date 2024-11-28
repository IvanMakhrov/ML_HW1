import requests
import json
import time
import pandas as pd
import io


class TestApi:
    def __init__(self):
        self.url_item = "http://127.0.0.1:8000/predict_item"
        self.url_items = "http://127.0.0.1:8000/predict_items"
        self.json_file, self.csv_file = 'test.json', 'test.csv'

    def test_json(self):
        file = open(f'service_files/{self.json_file}')
        data = [line for line in json.load(file)]

        i = 0
        while i != 5:
            response = requests.post(self.url_item, json=data[i])
            print(f'Car: {i+1}\nStatus Code: {response.status_code}\nPredicted price: {response.text}\n\n')
            time.sleep(0.5)
            i += 1

    def test_csv(self):

        with open(f'service_files/{self.csv_file}', 'rb') as file:
            file = {"file": file}
            response = requests.post(self.url_items, files=file).content
            df = pd.read_csv(io.StringIO(response.decode('utf-8')))
            print(df.head(5))
            print(df.info())


test_api = TestApi()
test_api.test_json()
test_api.test_csv()
