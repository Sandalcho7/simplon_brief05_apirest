import numpy as np
import pickle

from fastapi import FastAPI, HTTPException, Query
from enum import Enum

from utilities.config import MODELS_FOLDER, DTR_MODEL_FILE, RFR_MODEL_FILE



app = FastAPI()


with open(MODELS_FOLDER + DTR_MODEL_FILE, "rb") as model_dtr_file:
    model_dtr = pickle.load(model_dtr_file)

with open(MODELS_FOLDER + RFR_MODEL_FILE, "rb") as model_rfr_file:
    model_rfr = pickle.load(model_rfr_file)

@app.on_event("startup")
def load_models():
    global model_dtr, model_rfr



class TypeBatiment(str, Enum):
    Appartement = 'Appartement'
    Maison = 'Maison'

@app.get("/predict_dtr")
async def predict(
    longitude: float,
    latitude: float,
    vefa: bool,
    n_pieces: int,
    surface_habitable: int,
    type_Batiment: TypeBatiment = Query(...)
):
    try:
        if type_Batiment == TypeBatiment.Appartement:
            type_Appartement = 1
            type_Maison = 0
        else:
            type_Appartement = 0
            type_Maison = 1

        # Create a NumPy array from the input data
        input_array = np.array([[longitude, latitude, type_Appartement, type_Maison, vefa, n_pieces, surface_habitable]])

        # Use the loaded model to make predictions
        prediction = model_dtr.predict(input_array)

        # Return the prediction as JSON
        return {"prediction": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")


@app.get("/predict_rfr")
async def predict_rfr(
    longitude: float,
    latitude: float,
    vefa: bool,
    n_pieces: int,
    surface_habitable: int,
    type_Batiment: TypeBatiment = Query(...)
):
    try:
        if type_Batiment == TypeBatiment.Appartement:
            type_Appartement = 1
            type_Maison = 0
        else:
            type_Appartement = 0
            type_Maison = 1

        # Create a NumPy array from the input data
        input_array = np.array([[longitude, latitude, type_Appartement, type_Maison, vefa, n_pieces, surface_habitable]])

        # Use the loaded model to make predictions
        prediction = model_rfr.predict(input_array)

        # Return the prediction as JSON
        return {"prediction": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)