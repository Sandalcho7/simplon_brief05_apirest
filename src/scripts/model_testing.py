
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
FILTERED_CSV_FILE = os.getenv("FILTERED_CSV_FILE")


filtered_df = pd.read_csv(DATA_DIR + FILTERED_CSV_FILE)

# Features declaration
X = filtered_df[
    [
        "longitude",
        "latitude",
        "type_Appartement",
        "type_Maison",
        "n_pieces",
        "surface_habitable",
        "vefa",
    ]
].values
y = filtered_df["prix_m2"].values

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dataframe split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Different models test
params_grid = {
    "DTR": {
        "model": DecisionTreeRegressor(),
        "params": {"max_depth": list(range(1, 100))},
    },
    "KNR": {
        "model": KNeighborsRegressor(),
        "params": {"n_neighbors": list(range(1, 100))},
    },
    "LR": {
        "model": LinearRegression(),
        "params": {"fit_intercept": [True, False]},
    },
    "RFR": {
        "model": RandomForestRegressor(),
        "params": {
            "max_depth": list(range(20, 30, 5)),
            "min_samples_leaf": list(range(15, 25, 5)),
            "n_estimators": list(range(700, 900, 100)),
        },
    },
}

for model_name, model_config in params_grid.items():
    gs = GridSearchCV(
        estimator=model_config["model"], 
        param_grid=model_config["params"], 
        n_jobs=-1,
        error_score="raise"  # Debugging failing fits
    )
    gs.fit(X_train, y_train)
    print(
        f"{model_name} model with optimal params: {gs.best_params_} gives an MSE of"
    )
    print(np.sqrt(mean_squared_error(y_test, gs.best_estimator_.predict(X_test))))