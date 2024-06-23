import pytest
from httpx import AsyncClient
from app import app  # import your FastAPI instance

# defined an id_client for testting
id_client = "102833"


# Test the welcome endpoint
# Access Link: http://localhost:8000/
@pytest.mark.asyncio
async def test_welcome():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello world! Welcome to the Default Predictor API!"


# Test the /credit endpoint
# Access Link: http://localhost:8000/credit
@pytest.mark.asyncio
async def test_credit():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/credit")
    assert response.status_code == 200


# Test the /credit/{id_client}/predict endpoint
# Access Link: http://localhost:8000/credit/102833/predict
@pytest.mark.asyncio
async def test_predict_score_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(f"/credit/{id_client}/predict")
    assert response.status_code == 200


# Test the /credit/{id_client}/data endpoint
# Access Link: http://localhost:8000/credit/102833/data
@pytest.mark.asyncio
async def test_donnees_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(f"/credit/{id_client}/data")
    assert response.status_code == 200


# Test the /credit/{id_client}/shap endpoint
# Access Link: http://localhost:8000/credit/102833/shap
@pytest.mark.asyncio
async def test_shap_values_client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get(f"/credit/{id_client}/shap")
    assert response.status_code == 200


# To run a pytest for your test_api.py : pytest test_api.py
