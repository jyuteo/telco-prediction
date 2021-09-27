import io
import os
import sys
import json
import pytest

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)
sys.path.append(os.path.join(main_dir, "app"))
from app.main import create_app

app = create_app()


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def predict(client, request_body):
    return client.post("/predict", json=request_body)


# 1. test case: valid response body
# will return a list of prediction object if request body is valid
def test_for_valid_request_body(client):
    accepted_request_body = {
        "data": [{
            "CustomerID": "4808-GHDJN",
            "Gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 31.78,
            "TotalCharges": 31.78
        }, {
            "CustomerID": "4808-GHDJN",
            "Gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure": 2,
            "PhoneService": "No",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 31.78,
            "TotalCharges": 55.78
        }]
    }

    expected_response = {
        "predictions": [{
            "default": 0,
            "probability": 0.39704
        }, {
            "default": 0,
            "probability": 0.44063
        }]
    }

    response = predict(client, accepted_request_body)
    response_body = json.load(io.BytesIO(response.data))
    assert response.status_code == 200
    assert sorted(response_body) == sorted(expected_response)


# 2. test case: imcomplete keys in data object or wrong key name given
# will return error when data object is incomplete
def test_for_incomplete_key_in_request_body(client):
    # response boody without "CustomerID"
    invalid_request_body = {
        "data": [{
            "Gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 31.78,
            "TotalCharges": 31.78
        }]
    }

    expected_response_for_incomplete_keys = {
        "error": "key CustomerID is not found in data object index 0"
    }

    response = predict(client, invalid_request_body)
    response_body = json.load(io.BytesIO(response.data))
    assert response.status_code == 500
    assert sorted(response_body) == sorted(
        expected_response_for_incomplete_keys)


# 3. test case: wrong data type in data object for a key
# will return error when wrong data type is given for any key
def test_for_wrong_data_type_in_data_object(client):
    # invalid data for "Gender", accepted ["Female", "Male"], given "F"
    invalid_request_body = {
        "data": [{
            "CustomerID": "4808-GHDJN",
            "Gender": "F",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "Tenure": 2,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 31.78,
            "TotalCharges": 31.78
        }]
    }

    expected_response_for_invalid_data_type = {
        "error":
        "key Gender must be one of ['Female', 'Male'], but given F in data object index 0"
    }

    response = predict(client, invalid_request_body)
    response_body = json.load(io.BytesIO(response.data))
    assert response.status_code == 500
    assert sorted(response_body) == sorted(
        expected_response_for_invalid_data_type)
