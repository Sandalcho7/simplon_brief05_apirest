import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from config import FILTERED_CSV_PATH, MODELS_FOLDER, DTR_MODEL_FILE, RFR_MODEL_FILE



# Dataframe creation
filtered_df = pd.read_csv(FILTERED_CSV_PATH)

# Features declaration
X = filtered_df[['longitude', 'latitude', 'type_Appartement', 'type_Maison', 'n_pieces', 'surface_habitable', 'vefa']].values
y = filtered_df['prix_m2'].values

# Dataframe split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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



if not os.path.exists("../" + MODELS_FOLDER):
    os.makedirs("../" + MODELS_FOLDER)


# Training a model from the list above
training(model_dtr)

# Export
with open("../" + MODELS_FOLDER + DTR_MODEL_FILE, 'wb') as file:
    pickle.dump(model_dtr, file)


# Training a model from the list above
training(model_rfr)

# Export
with open("../" + MODELS_FOLDER + RFR_MODEL_FILE, 'wb') as file:
    pickle.dump(model_rfr, file)