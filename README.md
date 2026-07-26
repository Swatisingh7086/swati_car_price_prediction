# Car Price Predictor (CarDekho dataset)

A Flask web app that predicts used car prices using a Linear Regression model,
retrained on the CarDekho dataset (`CAR_DETAILS_FROM_CAR_DEKHO.csv`).

## Project structure

```
├── application.py              # Flask app (run this to start the server)
├── model_utils.py               # Shared helper needed to load the pickled model
├── train_model.py                # Script that cleans data + trains + saves the model
├── CAR_DETAILS_FROM_CAR_DEKHO.csv  # Raw source data
├── Cleaned_Car_data.csv          # Cleaned data (auto-generated, used by the app)
├── LinearRegressionModel.pkl     # Trained model (auto-generated, used by the app)
├── requirements.txt
├── Procfile                      # For deployment (Heroku/Render, gunicorn)
├── templates/
│   └── index.html                # Web form UI
└── static/
    └── css/
        └── style.css