import os
import traceback
import numpy as np

from flask import Flask, request, jsonify
from joblib import load
from dotenv import load_dotenv
from validate import validate_request_body
from utils import prepare_df

app = Flask(__name__)
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
print("loading model...")
model = load(MODEL_PATH)
print("loaded: {}\n".format(MODEL_PATH))


def create_app():
    app = Flask(__name__)

    @app.errorhandler(Exception)
    def handle_exception(err):
        app.logger.error("Unknown Exception: {}".format(str(err)))
        app.logger.debug(''.join(
            traceback.format_exception(etype=type(err),
                                       value=err,
                                       tb=err.__traceback__)))
        response = {"error": str(err)}
        return jsonify(response), 500

    @app.route('/')
    def base():
        return "use endpoint /predict for prediction"

    @app.route('/predict', methods=['POST'])
    def predict():
        if model:
            json_body = request.json
            validation_result = validate_request_body(json_body)
            if not validation_result["status"]:
                raise Exception(validation_result["message"])

            data_ls = validation_result["data_ls"]
            processed_df = prepare_df(data_ls)

            probabilities = np.round(model.predict_proba(processed_df), 5)
            probability_for_deafult = list(map(float, probabilities[:, 1]))
            predicted_class = list(map(int, np.argmax(probabilities, axis=1)))

            predictions = list()
            for i in range(len(predicted_class)):
                predictions.append({
                    "default": predicted_class[i],
                    "probability": probability_for_deafult[i],
                })
            return jsonify({"predictions": predictions})

        else:
            raise Exception("Unable to load model")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", debug=True)
