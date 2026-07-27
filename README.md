# 🚗 Car Price Prediction

A Machine Learning project that predicts the selling price of a used car using **Linear Regression**. The model is trained on the CarDekho dataset and deployed with **Flask**.

## 📌 Features

* Predicts used car prices based on car details.
* Built using Linear Regression.
* Data preprocessing with Scikit-learn Pipeline.
* Handles categorical features using OneHotEncoder.
* Model saved using Pickle for quick loading.

## 🛠️ Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* Pickle

## 📂 Project Structure

```text
Car-Price-Prediction/
│── application.py
│── LinearRegressionModel.pkl
│── Cleaned_Car_data.csv
│── model_utils.py
│── requirements.txt
│── Procfile
│── README.md
│── .gitignore
```

## 📊 Dataset

This project uses the **CarDekho Used Car Dataset**.

**Input Features**

* Company
* Car Name
* Year
* Fuel Type
* Seller Type
* Transmission
* Owner
* Kilometers Driven

**Target**

* Selling Price

## ⚙️ Machine Learning Workflow

1. Data Cleaning
2. Data Preprocessing
3. Train-Test Split
4. Feature Encoding using OneHotEncoder
5. Model Training with Linear Regression
6. Model Saving using Pickle

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Car-Price-Prediction.git
```

Go to the project directory:

```bash
cd Car-Price-Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python application.py
```

The application will start on:

```text
http://127.0.0.1:5000/
```

## 📈 Model

* **Algorithm:** Linear Regression
* **Preprocessing:** Scikit-learn Pipeline
* **Model File:** `LinearRegressionModel.pkl`

## 💡 Future Improvements

* Improve prediction accuracy using ensemble models such as Random Forest or XGBoost.
* Add model evaluation metrics.
* Deploy the application on Render or another cloud platform.
* Build a REST API for predictions.

## 👩‍💻 Author

**Swati Singh**

GitHub: https://github.com/Swatisingh7086

LinkedIn:https://www.linkedin.com/in/swati-singh-01b70141b
