# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():

#прочитайте файл с гиперпараметрами params.yaml

    with open('params.yaml', 'r') as fd:
        params=yaml.safe_load(fd)

#загрузите результат прошлого шага: fitted_model.pkl
    data=pd.read_csv('data/initial_data.csv')

    with open('models/fitted_model.pkl', 'rb') as fd:
        model=joblib.load(fd)
    X = data.drop('target', axis=1)
    y = data['target']

#реализуйте основную логику шага с использованием прочтённых гиперпараметров

    cv_strategy=StratifiedKFold(n_splits=params.get('n_splits', 5))
    cv=cross_validate(model, X, y, cv=cv_strategy, scoring=['f1', 'roc_auc'])

    os.makedirs('cv_results', exist_ok=True)

    cv_res = {
    'fit_time': float(cv['fit_time'].mean().round(2)),
    'score_time':float(cv['score_time'].mean().round(2)),
    'test_f1': float(cv['test_f1'].mean().round(2)),
    'test_roc_auc': float(cv['test_roc_auc'].mean().round(2))
    }

#сохраните результата кросс-валидации в cv_res.json

    with open('cv_results/cv_res.json', 'w') as fd:
        json.dump(cv_res, fd)

if __name__ == '__main__':
    evaluate_model()