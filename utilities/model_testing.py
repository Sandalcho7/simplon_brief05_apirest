import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from config import FILTERED_CSV_PATH



# Dataframe creation
filtered_df = pd.read_csv(FILTERED_CSV_PATH)

# Features declaration
X = filtered_df[['longitude', 'latitude', 'type_Appartement', 'type_Maison', 'n_pieces', 'surface_habitable', 'vefa']].values
y = filtered_df['prix_m2'].values

# Dataframe split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Different models test
params_grid = {
                'DTR': {
                    'model': DecisionTreeRegressor(),
                    'params': {
                        'max_depth': list(range(1, 100)),
                    }
                },
                'KNR': {
                    'model': KNeighborsRegressor(),
                    'params': {
                        'n_neighbors': list(range(1, 100))
                    }
                },
                'LR': {
                    'model': LinearRegression(),
                    'params': {
                        'fit_intercept': [True, False],
                        'positive': [True, False]
                    }
                },
                'RFR': {
                    'model': RandomForestRegressor(),
                    'params': {
                        'max_depth': list(range(20, 30, 5)),
                        'min_samples_leaf': list(range(15, 25, 5)),
                        'n_estimators': list(range(700, 900, 100))
                    }
                }
            }

for model_name, model_config in params_grid.items():
    gs = GridSearchCV(estimator=model_config['model'], 
                      param_grid=model_config['params'], n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f'Modèle: {model_name} avec params optimaux: {gs.best_params_} donne erreur =')
    print(np.sqrt(mean_squared_error(y_test, gs.best_estimator_.predict(X_test))))