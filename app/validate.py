schema = {
    "CustomerID": {"type": str},
    "Gender": {"type": "categorical", "accepted": ["Female", "Male"]},
    "SeniorCitizen": {"type": "categorical", "accepted": [0, 1]},
    "Partner": {"type": "categorical", "accepted": ["Yes", "No"]},
    "Dependents": {"type": "categorical", "accepted": ["Yes", "No"]},
    "Tenure": {"type": int},
    "PhoneService": {"type": "categorical", "accepted": ["Yes", "No"]},
    "MultipleLines": {"type": "categorical", "accepted": ["Yes", "No", "No phone service"]},
    "InternetService": {"type": "categorical", "accepted": ["Fiber optic", "DSL", "No"]},
    "OnlineSecurity": {"type": "categorical", "accepted": ["Yes", "No", "No internet service"]},
    "OnlineBackup": {"type": "categorical", "accepted": ["Yes", "No", "No internet service"]},
    "DeviceProtection": {"type": "categorical", "accepted": ["Yes", "No", "No internet service"]},
    "TechSupport": {"type": "categorical", "accepted": ["Yes", "No", "No internet service"]},
    "StreamingTV": {"type": "categorical", "accepted": ["Yes", "No", "No internet service"]},
    "StreamingMovies": {"type": "categorical", "accepted": ["Yes", "No", "No internet service"]},
    "Contract": {"type": "categorical", "accepted": ["Month-to-month", "Two year", "One year"]},
    "PaperlessBilling": {"type": "categorical", "accepted": ["Yes", "No"]},
    "PaymentMethod": {"type": "categorical", "accepted": ["Electronic check", "Mailed check",
                                                          "Bank transfer (automatic)", "Credit card (automatic)"]},
    "MonthlyCharges": {"type": float},
    "TotalCharges": {"type": float}
}


def categorical_error_message(key, accepted, value, i):
    return "key {} must be one of {}, but given {} in data object index {}".format(key, accepted, value, i)


def type_error_message(key, accepted, value, i):
    return "key {} must be of type {}, but given type {}:{} in data object index {}".format(key, accepted, type(value), value, i)


def validate_data(data, i):
    new_data = dict()
    for key in schema:
        if key not in data:
            return {"status": False, "message": "key {} is not found in data object index {}".format(key, i)}
        if schema[key]["type"] == "categorical":
            if data[key] not in schema[key]["accepted"]:
                return {"status": False, "message": categorical_error_message(
                    key, schema[key]["accepted"], data[key], i)}
        else:
            if not isinstance(data[key], schema[key]["type"]):
                return {"status": False, "message": type_error_message(key, schema[key]["type"], data[key], i)}
        new_data[key] = data[key]
    return {"status": True, "data": new_data}


def validate_request_body(json_body):
    if "data" not in json_body:
        return {"status": False, "message": "key \"data\" not found in request body"}

    data_ls = json_body["data"]
    if not isinstance(data_ls, list):
        return {"status": False, "message": "data given is not a list of objects"}
    if len(data_ls) < 1:
        return {"status": False, "message": "no data was given"}

    new_data_ls = list()
    for i, data in enumerate(data_ls):
        validate_result = validate_data(data, i)
        if not validate_result["status"]:
            return validate_result
        new_data_ls.append(validate_result["data"])
    return {"status": True, "data_ls": new_data_ls}
