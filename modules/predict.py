import logging
import glob
import dill
import os
import json
from datetime import datetime
import pandas as pd


# Объявление переменных для навигации
def path_variables():
    if __name__ == '__main__':
        path = os.environ.get('PROJECT_PATH', str(os.getcwd())[:-8])
    else:
        path = f'{str(os.getcwd())}/airflow_hw'
    logging.info(f'__name__ is {__name__}')
    logging.info(f'Path is set as {path}')
    list_of_models = glob.glob(f'{path}/data/models/*.pkl')
    latest_model_path = str(max(list_of_models, key=os.path.getctime))
    json_files = [f'{path}/data/test/{pos_json}' for pos_json in os.listdir(f'{path}/data/test/') if pos_json.endswith('.json')]
    return path, latest_model_path, json_files


# Функция, подгружающая тестовые входные данные для модели
def load_jsons(json_files):
    json_data = pd.DataFrame(columns=['description', 'fuel', 'id', 'image_url', 'lat', 'long', 'manufacturer', 'model', 'odometer', 'posting_date', 'price', 'region', 'region_url', 'state', 'title_status', 'transmission', 'url', 'year'])
    i = 0
    for filepath in json_files:
        file = open(filepath,)
        entry = pd.DataFrame(json.load(file), index=[i])
        json_data = pd.concat([json_data, entry])
        i += 1
    return json_data


# Осуществление предсказаний
def predict():
    path, latest_model_path, json_files = path_variables()
    with open(latest_model_path, 'rb') as pkl_model:
        model = dill.load(pkl_model)
    json_data = load_jsons(json_files)
    predictions = pd.Series(model.predict(json_data))
    result = pd.concat([json_data['id'], predictions], axis=1)
    result.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}')


if __name__ == '__main__':
    predict()
