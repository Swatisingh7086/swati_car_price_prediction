import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv('Cleaned_Car_data.csv')
print(df.columns.tolist())
print(df.head())
print("Shape:", df.shape)
df.head()

df.info()

df.describe(include='all').T

print("Missing values per column:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
print(df.columns.tolist())
if 'kms_driven' in df.columns:
    df.rename(columns={'kms_driven': 'km_driven'}, inplace=True)
for col in ['km_driven', 'Price']:
    if col in df.columns and df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled missing values in '{col}' with median: {median_val}")

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
after = len(df)
print(f"Removed {before - after} duplicate rows. New shape: {df.shape}")

if 'name' in df.columns:
    df['brand'] = df['name'].astype(str).str.split().str[0].str.title()
    brand_counts = df['brand'].value_counts()
    rare_brands = brand_counts[brand_counts < 10].index
    df['brand'] = df['brand'].replace(rare_brands, 'Other')
    df.drop(columns=['name'], inplace=True)
    print(df['brand'].value_counts())
else:
    print("No 'name' column found — skipping brand extraction.")

plt.figure()
sns.boxplot(x=df['km_driven'])
plt.title("km_driven — before capping outliers")
plt.show()
cap_value = df['km_driven'].quantile(0.99)
n_capped = (df['km_driven'] > cap_value).sum()
df['km_driven'] = np.where(df['km_driven'] > cap_value, cap_value, df['km_driven'])
print(f"Capped {n_capped} rows at km_driven = {cap_value:.0f}")
plt.figure()
sns.boxplot(x=df['km_driven'])
plt.title("km_driven — after capping outliers")
plt.show()

CURRENT_YEAR = 2026
df['car_age'] = CURRENT_YEAR - df['year']
df.drop(columns=['year'], inplace=True)
df[['car_age']].describe()

plt.figure()
#sns.histplot(df['Price'], kde=True, bins=40)

sns.histplot(df['Price'], kde=True, bins=40)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.show()

numeric_cols = ['car_age', 'km_driven', 'Price']
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap — Numeric Features vs Selling Price")
plt.tight_layout()
plt.savefig("../Images/correlation_heatmap.png", dpi=150)
plt.show()
corr_matrix

Price_corr = corr_matrix['Price'].drop('Price').sort_values(key=abs, ascending=False)
print("Correlation with Price, ranked by strength:")
Price_corr

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df.groupby('fuel')['Price'].mean().sort_values().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title("Average Selling Price by Fuel Type")
axes[0].set_ylabel("Average Selling Price")
df.groupby('transmission')['Price'].mean().sort_values().plot(kind='bar', ax=axes[1], color='seagreen')
axes[1].set_title("Average Selling Price by Transmission")
axes[1].set_ylabel("Average Selling Price")
plt.tight_layout()
plt.savefig("../Images/avg_Price_by_fuel_transmission.png", dpi=150)
plt.show()

plt.figure()
sns.scatterplot(data=df, x='km_driven', y='Price', alpha=0.5)
plt.title("km_driven vs Selling Price")
plt.savefig("../Images/scatter_km_driven_vs_Price.png", dpi=150)
plt.show()

skewness = df['Price'].skew()
print(f"Skewness of Price: {skewness:.2f}")
USE_LOG_TARGET = abs(skewness) > 1
print("Using log-transformed target:" , USE_LOG_TARGET)
if USE_LOG_TARGET:
    df['Price_log'] = np.log1p(df['Price'])

categorical_cols = [c for c in ['fuel', 'seller_type', 'transmission', 'owner', 'brand'] if c in df.columns]
print("Encoding:", categorical_cols)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded.head()

target_col = 'Price_log' if USE_LOG_TARGET else 'Price'
drop_cols = ['Price', 'Price_log'] if USE_LOG_TARGET else ['Price']
X = df_encoded.drop(columns=[c for c in drop_cols if c in df_encoded.columns])
y = df_encoded[target_col]
print("Features:", X.shape, " Target:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained.")

y_pred = model.predict(X_test)
if USE_LOG_TARGET:
    y_test_eval = np.expm1(y_test)
    y_pred_eval = np.expm1(y_pred)
else:
    y_test_eval = y_test
    y_pred_eval = y_pred

r2 = r2_score(y_test_eval, y_pred_eval)
mae = mean_absolute_error(y_test_eval, y_pred_eval)
rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred_eval))
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:,.2f}")
print(f"RMSE     : {rmse:,.2f}")

plt.figure()
plt.scatter(y_test_eval, y_pred_eval, alpha=0.5)
lims = [min(y_test_eval.min(), y_pred_eval.min()), max(y_test_eval.max(), y_pred_eval.max())]
plt.plot(lims, lims, 'r--', label='Perfect Prediction')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Predicted vs Actual Selling Price")
plt.legend()
plt.savefig("../Images/predicted_vs_actual.png", dpi=150)
plt.show()

summary_corr = Price_corr.to_frame(name='correlation_with_Price')
summary_corr
