# Docker container running a Flask API for telco data classfication

Predicts the likelihood of customer defaulting on telco payment based on their telco data

## 1. Endpoint `/predict`

### Request body

JSON object with key `"data"` a list of data objects as values

```json
{
    "data": [
        {
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
        },
        ...
    ]
}
```

### Response body

A list of predictions object with predicted class and probablity

```json
{
  "predictions": [
    {
      "default": 0,
      "probability": 0.39704
    },
    ...
  ]
}
```

## 2. Running with Docker

### Details:

| Name         |            Using             | Description                                                      |
| ------------ | :--------------------------: | :--------------------------------------------------------------- |
| `FLASK_HOST` |          `0.0.0.0`           | hostname on which flask will run                                 |
| `FLASK_PORT` |            `5000`            | port on which flask will run                                     |
| `MODEL_PATH` | `../model/model_name.joblib` | model file path, relative to `/app`, can be configured in `.env` |

1. Configure the model path in `.env`

   ```
   MODEL_PATH=../model/stacking-gbs-lr-rf.joblib
   ```

2. Build docker container, run:

   ```
   docker-compose up
   ```

3. To stop, run:

   ```
   docker-compose down
   ```

## 3. Running locally

1. Configure the model path in `.env`

   ```
   MODEL_PATH=../model/stacking-gbs-lr-rf.joblib
   ```

2. Create a virtual environment and install required packages

   ```
   pip install -r requirements.txt
   ```

3. In `/app`, run:

   ```
   python main.py
   ```

## 4. API Tests

To run tests in `/tests`, run:

```
pytest -v
```

## 5. Using the API

The script `/inference/inference.py` takes in a `.csv` file of data to be inferenced. Its columns should match all the keys of the data object.

Run:

```python
python inference.py --i "./data/dummy_data.csv"
```

It will call the `/predict` endpoint and return a `.csv` file of the classification results
| | CustomerID | Default | Probability |
| --- | ------ | ------- | ----------- |
| 0 | 7590-VHVEG | 1 | 0.83529 |
| 1 | 5575-GNVDE | 0 | 0.08385 |
| 2 | 6388-TABGU | 0 | 0.02351 |

## 6. Model information

Trained with Scikit-learn models, with 5 folds cross validation.

The model performance are listed in the table below.

The best model (highest average f1 score across 5 folds) is saved in `/model`

To train models, in `/train`, run:

```python
python train.py -i "./data/raw.csv"
```

### Performance

| Model                             | Mean accuracy | Mean precision | Mean recall | Mean f1 | Time    |
| --------------------------------- | ------------- | -------------- | ----------- | ------- | ------- |
| stacking-gbs-lr-rf                | 0.77616       | 0.5595         | 0.7469      | 0.6396  | 3015.57 |
| logistic-regression               | 0.7792        | 0.5657         | 0.7303      | 0.6374  | 8.23    |
| gradient-boosting-smote           | 0.7745        | 0.5584         | 0.7333      | 0.6334  | 715.45  |
| mlp-oversampling                  | 0.7554        | 0.5266         | 0.7947      | 0.6333  | 573.62  |
| logistic-regression-undersampling | 0.7518        | 0.5219         | 0.8049      | 0.6330  | 6.34    |
| random-forest                     | 0.7875        | 0.5856         | 0.6870      | 0.6321  | 97.82   |
| logistic-regression-smote         | 0.7511        | 0.5210         | 0.8039      | 0.6320  | 14.82   |
| mlp-smote                         | 0.7580        | 0.5309         | 0.7799      | 0.6314  | 356.90  |
| gradient-boosting-oversampling    | 0.7530        | 0.5238         | 0.7931      | 0.6307  | 573.33  |
| logistic-regression-oversampling  | 0.7479        | 0.5168         | 0.8097      | 0.6307  | 8.72    |
| gradient-boosting-undersampling   | 0.7521        | 0.5225         | 0.7914      | 0.6293  | 217.78  |
| random-forest-smote               | 0.7445        | 0.5127         | 0.7992      | 0.6242  | 140.31  |
| random-forest-oversampling        | 0.7258        | 0.4911         | 0.8334      | 0.6177  | 122.98  |
| random-forest-undersampling       | 0.7267        | 0.4918         | 0.8291      | 0.6172  | 68.70   |
| gradient-boosting                 | 0.8038        | 0.6643         | 0.5302      | 0.5897  | 395.29  |

### Explanation

- The problem faced is class imbalance in the training data. So, accuracy is not a good metric to determine the model performance. F1 score is used to select the best model by balancing precision and recall
- Resampling methods such as random oversampling, random undersampling and SMOTE are used prior to model fitting
- Hyperparameter tuning was done for each model, although with no significance increase in the metrics
- The model with highest f1 score is Stacking Classifier using Logistic Regression, Random Forest and Gradient Boosting
