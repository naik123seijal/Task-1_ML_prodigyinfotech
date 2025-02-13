import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
data = {
'Square_Footage': np.random.randint(500, 4000, 100),
'Bedrooms': np.random.randint(1, 6, 100),
'Bathrooms': np.random.randint(1, 4, 100),
'Price': np.random.randint(100000, 1000000, 100)
}
df = pd.DataFrame(data)

print("Dataset Preview:")
print(df.head())
X = df[['Square_Footage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
