import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import yaml
import os
import joblib

def fit_model():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    data = pd.read_csv('data/initial_data.csv')
    target_col = params.get('target_col', 'target')
    X = data.drop(columns=[target_col])
    y = data[target_col]

    cat_features = X.select_dtypes(include='object')
    num_features = X.select_dtypes(['float'])

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop=params.get('one_hot_drop', 'if_binary')), cat_features.columns.tolist()),
        ('num', StandardScaler(), num_features.columns.tolist())
    ], remainder='drop', verbose_feature_names_out=False)

    model = LogisticRegression(
        penalty=params.get('penalty', 'l2'),
        C=params.get('C', 1.0)
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X, y)

    os.makedirs('models', exist_ok=True)
    with open('models/fitted_model.pkl', 'wb') as fd:
        joblib.dump(pipeline, fd)

if __name__ == '__main__':
    fit_model()
