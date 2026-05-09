import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('car data.csv')
df.head(2)

df.info()

print(df['Car_Name'].unique())

current_year = pd.Timestamp.now().year
df['Car_Age'] = current_year - df['Year']

df['Car_Rank'] = df.groupby('Car_Name')['Selling_Price'].transform('mean')

X = df.drop(['Selling_Price', 'Car_Name', 'Year'], axis=1)
y = df['Selling_Price']

X.head(2)

numeric_features = ['Car_Rank', 'Car_Age', 'Present_Price', 'Driven_kms', 'Owner']
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']

numeric_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=600,learning_rate=0.03,random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipeline.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=10,
    scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-scores)

print("RMSE:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("STD RMSE:", rmse_scores.std())

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)