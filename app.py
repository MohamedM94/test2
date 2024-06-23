import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List

app = FastAPI()

# Define Pydantic models for responses


class ListResponseModel(BaseModel):
    liste_id: List[int]
    liste_features: List[str]


class ProbabilityResponseModel(BaseModel):
    probability: float


# Define a Pydantic model for your response
class ItemResponse(BaseModel):
    liste_id: List[int]
    liste_features: List[str]
    idx_client: List[int]


class DataResponseModel(BaseModel):
    data: List[dict]


class ShapValuesResponseModel(BaseModel):
    shap_val: List[float]


# Load data and model when needed

data_test = pd.read_csv("donnees_test_essai.csv")
data_train = pd.read_csv("donnees_train_essai.csv")
group_0 = data_train[data_train["TARGET"] == 0]
group_1 = data_train[data_train["TARGET"] == 1]

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

unwrapped_model = model.named_steps["classifier"]
explainer = shap.TreeExplainer(unwrapped_model)

# liste_id = data_test["SK_ID_CURR"].to_list()
# idx_client = data_test.index[data_test["SK_ID_CURR"] == liste_id].to_list()
# liste_features = data_test.columns.tolist()


def remove_sk_id_curr(data):
    return data.drop(columns=["SK_ID_CURR"])


# test local : http://127.0.0.1:8000/
@app.get("/")
def welcome():
    return "Hello world! Welcome to the Default Predictor API!"


# test local : http://127.0.0.1:8000/credit/
@app.get("/credit", response_model=ItemResponse)
def liste_identifiants():
    # Check if columns exist
    if "SK_ID_CURR" not in data_test.columns:
        raise HTTPException(status_code=400, detail="SK_ID_CURR not in data")

    liste_features = data_test.columns.tolist()
    liste_id = data_test["SK_ID_CURR"].tolist()
    idx_client_list = data_test.index.tolist()  # get index of all rows

    return {
        "liste_id": liste_id,
        "liste_features": liste_features,
        "idx_client": idx_client_list,  # return the entire list
    }


print(data_test)
liste_id = data_test["SK_ID_CURR"].to_list()
print(liste_id)
liste_features = data_test.columns.tolist()
print(liste_features)


# test local : http://127.0.0.1:8000/credit/425013/predict
@app.get("/credit/{id_client}/predict", response_model=ProbabilityResponseModel)
def predict_score_client(id_client: int):
    if id_client in liste_id:
        data_client = data_test.loc[data_test["SK_ID_CURR"] == id_client]
        proba = model.predict_proba(data_client)
        proba_0 = round(proba[0][0] * 100)
        return {"probability": proba_0}
    else:
        raise HTTPException(status_code=404, detail="Unknown ID")


# test local : http://127.0.0.1:8000/credit/425013/data
@app.get("/credit/{id_client}/data", response_model=DataResponseModel)
def donnees_client(id_client: int):
    data_client = data_test.loc[data_test["SK_ID_CURR"] == id_client]
    return {"data": data_client.to_dict(orient="records")}


# test local : http://127.0.0.1:8000/credit/425013/shap
@app.get("/credit/{id_client}/shap", response_model=ShapValuesResponseModel)
def shap_values_client(id_client: int):
    if id_client in liste_id:
        data_client = data_test.loc[data_test["SK_ID_CURR"] == id_client]
        data_client = remove_sk_id_curr(data_client)
        shap_values = explainer.shap_values(data_client)
        shap_data_flat = [float(val) for val in shap_values[0].ravel()]
        return {"shap_val": shap_data_flat}  # corrected line
    else:
        raise HTTPException(status_code=404, detail="Unknown ID")


# to run the app in local : uvicorn app:app --reload
# to check the automatic interactive FastAPI documentation : http://127.0.0.1:8000/docs
