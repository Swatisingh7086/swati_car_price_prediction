import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

os.makedirs("Images", exist_ok=True)


df = pd.read_csv('CAR_DETAILS_FROM_CAR_DEKHO.csv')


print("Shape of Dataset:", df.shape)
df.to_csv("Cleaned_Car_data.csv", index=False)
print("Cleaned dataset saved successfully.")

print(df.head())

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe(include="all").T)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nColumn Names")
print(df.columns.tolist())
if "kms_driven" in df.columns:
    df.rename(columns={"kms_driven": "km_driven"}, inplace=True)

if "selling_price" not in df.columns:
    if "price" in df.columns:
        df.rename(columns={"price": "selling_price"}, inplace=True)
    elif "selling_price" not in df.columns and "Selling_Price" in df.columns:
        df.rename(columns={"Selling_Price": "selling_price"}, inplace=True)

numeric_columns = ["km_driven", "selling_price"]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].fillna(df[col].median(), inplace=True)

before = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("Duplicates Removed :", before - len(df))
print("New Shape :", df.shape)

if "name" in df.columns:
    df["brand"] = df["name"].apply(lambda x: str(x).split()[0])
    df["model"] = df["name"].apply(lambda x: " ".join(str(x).split()[1:]))

elif "car_name" in df.columns:
    df["brand"] = df["car_name"].apply(lambda x: str(x).split()[0])
    df["model"] = df["car_name"].apply(lambda x: " ".join(str(x).split()[1:]))

brand_count = df["brand"].value_counts()

rare_brand = brand_count[brand_count < 10].index

df["brand"] = df["brand"].replace(rare_brand, "Other")

if "name" in df.columns:
    df.drop("name", axis=1, inplace=True)

if "car_name" in df.columns:
    df.drop("car_name", axis=1, inplace=True)

df.to_csv("Cleaned_Car_data.csv", index=False)
print("Cleaned dataset saved successfully.")

print(df.head())
plt.figure(figsize=(8,5))
sns.boxplot(x=df["km_driven"])
plt.title("KM Driven Before Outlier Treatment")
plt.show()

cap = df["km_driven"].quantile(0.99)

df["km_driven"] = np.where(
    df["km_driven"] > cap,
    cap,
    df["km_driven"]
)

plt.figure(figsize=(8,5))
sns.boxplot(x=df["km_driven"])
plt.title("KM Driven After Outlier Treatment")
plt.show()
CURRENT_YEAR = 2026

df["car_age"] = CURRENT_YEAR - df["year"]

df.drop("year", axis=1, inplace=True)

print(df[["car_age"]].describe())
plt.figure(figsize=(8,5))
sns.histplot(df["selling_price"], bins=40, kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("Images/selling_price_distribution.png")
plt.show()


numeric_columns = ["selling_price", "km_driven", "car_age"]

if "present_price" in df.columns:
    numeric_columns.append("present_price")

corr = df[numeric_columns].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("Images/correlation_heatmap.png")
plt.show()


if "fuel" in df.columns:
    fuel_col = "fuel"
elif "fuel_type" in df.columns:
    fuel_col = "fuel_type"

plt.figure(figsize=(8,5))
df.groupby(fuel_col)["selling_price"].mean().sort_values().plot(kind="bar")
plt.title("Average Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Selling Price")
plt.tight_layout()
plt.savefig("Images/fuel_price.png")
plt.show()


plt.figure(figsize=(8,5))
df.groupby("transmission")["selling_price"].mean().sort_values().plot(kind="bar")
plt.title("Average Selling Price by Transmission")
plt.xlabel("Transmission")
plt.ylabel("Average Selling Price")
plt.tight_layout()
plt.savefig("Images/transmission_price.png")
plt.show()


plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="km_driven",
    y="selling_price",
    alpha=0.6
)
plt.title("KM Driven vs Selling Price")
plt.tight_layout()
plt.savefig("Images/km_vs_price.png")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x=df["selling_price"])
plt.title("Selling Price Boxplot")
plt.tight_layout()
plt.savefig("Images/price_boxplot.png")
plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x=df["car_age"])
plt.title("Car Age Boxplot")
plt.tight_layout()
plt.savefig("Images/car_age_boxplot.png")
plt.show()


print(df[numeric_columns].corr()["selling_price"].sort_values(ascending=False))
categorical_columns = []

for col in [
    "fuel",
    "fuel_type",
    "seller_type",
    "transmission",
    "owner",
    "brand",
    "model"
]:
    if col in df.columns:
        categorical_columns.append(col)

df_encoded = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

X = df_encoded.drop("selling_price", axis=1)
y = df_encoded["selling_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Shape :", X_train.shape)
print("Testing Shape :", X_test.shape)

model = LinearRegression()

model.fit(X_train, y_train)

print("Model Training Completed")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("----------------------------")
print("R2 Score :", round(r2, 4))
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")

plt.tight_layout()
plt.savefig("Images/predicted_vs_actual.png")
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(8,5))

sns.histplot(
    residuals,
    kde=True,
    bins=30
)

plt.title("Residual Distribution")

plt.tight_layout()
plt.savefig("Images/residual_distribution.png")

plt.show()

coef = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coef = coef.sort_values(
    by="Coefficient",
    key=abs,
    ascending=False
)

print("\nTop 15 Important Features")
print(coef.head(15))

coef.head(15).plot(
    x="Feature",
    y="Coefficient",
    kind="bar",
    figsize=(12,6),
    legend=False
)

plt.xticks(rotation=90)

plt.title("Top 15 Feature Importance")

plt.tight_layout()

plt.savefig("Images/feature_importance.png")

plt.show()
import pickle

with open("LinearRegressionModel.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully.")

predictions = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print(predictions.head(10))

predictions.to_csv("predictions.csv", index=False)

print("Predictions saved successfully.")

print("\nTraining Completed Successfully!")
print(f"R2 Score  : {r2:.4f}")
print(f"MAE       : {mae:.2f}")
print(f"RMSE      : {rmse:.2f}")
pickle.dump(X.columns.tolist(), open("model_columns.pkl", "wb"))

print("Feature columns saved successfully.")

import os

print(os.getcwd())