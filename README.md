# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and prepare data – Import dataset, select predictor variables (enginesize, horsepower, citympg, highwaympg) and target (price).
2. Split dataset – Divide into training and testing sets using train_test_split.
3. Scale features – Standardize predictors with StandardScaler for better regression performance.
4. Train model & predict – Fit LinearRegression on training data, generate predictions on test data.
5. Evaluate & visualize – Compute metrics (MSE, RMSE, R², MAE), check residuals (Durbin‑Watson, homoscedasticity), and plot actual vs predicted prices.

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df=pd.read_csv('CarPrice_Assignment.csv')

x=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred=model.predict(x_test_scaled)

print('Name: Suruthika')
print('Reg no: 212225040441')
print("MODEL COEFFICIENTS: ")
for feature,coef in zip(x.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10}")
print(f"{'intercept':>12}: {model.intercept_:10}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test,y_pred):>10}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test,y_pred)):>10}")
print(f"{'R-squared':>12}: {r2_score(y_test,y_pred):>10}")
print(f"{'ABSOLUTE':>12}: {mean_absolute_error(y_test,y_pred):>10}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(),y.max()],'r--')
plt.title("Linearity check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

residuals= y_test - y_pred
dw_test=sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistic: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10,5))
sns.residplot(x=y_pred,y=residuals,lowess=True,line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
   Developed by:V SURUTHIKA 
   RegisterNumber:  212225040441*/
```

##  Output:
<img width="342" height="185" alt="Screenshot 2026-02-04 140207" src="https://github.com/user-attachments/assets/6b968b82-2111-44f8-9e54-8d0bc6c50277" />

<img width="332" height="134" alt="Screenshot 2026-02-04 140214" src="https://github.com/user-attachments/assets/f60f3c56-0e62-42ef-837a-7a5395ab711b" />

<img width="1228" height="607" alt="Screenshot 2026-02-04 140231" src="https://github.com/user-attachments/assets/f917769d-469d-458b-ae4f-350f633839f4" />

<img width="578" height="61" alt="Screenshot 2026-02-04 140240" src="https://github.com/user-attachments/assets/64abc26b-9630-45be-9520-ecd066cb6716" />

<img width="1193" height="602" alt="Screenshot 2026-02-04 140257" src="https://github.com/user-attachments/assets/69c1dac6-4d84-46a7-9678-36661f6e3fdd" />

<img width="1277" height="525" alt="Screenshot 2026-02-04 140316" src="https://github.com/user-attachments/assets/b21dfdec-8d4f-49a3-8a07-da8086a6d7bb" />



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
