import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
FILTERED_CSV_FILE = os.getenv("FILTERED_CSV_FILE")
MODELS_DIR = os.getenv("MODELS_DIR")
DTR_MODEL_FILE = os.getenv("DTR_MODEL_FILE")
RFR_MODEL_FILE = os.getenv("RFR_MODEL_FILE")


# Dataframe creation
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

# Training function
def training(model):
    model.fit(X_train, y_train)

    train_error = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_error = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    print("Squared error on train data: " + str(train_error))
    print("Squared error on test data: " + str(test_error))


# Model list
model_dtr = DecisionTreeRegressor(max_depth=3)
model_rfr = RandomForestRegressor(max_depth=20, min_samples_leaf=15, n_estimators=700)


if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


# Training and export the DTR model
training(model_dtr)
with open(MODELS_DIR + DTR_MODEL_FILE, "wb") as file:
    pickle.dump(model_dtr, file)


# Training and export the RFR model
training(model_rfr)
with open(MODELS_DIR + RFR_MODEL_FILE, "wb") as file:
    pickle.dump(model_rfr, file)
