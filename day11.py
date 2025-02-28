from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values
model = LinearRegression()
# model = KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)

X_new = [[31721.3]]
print(model.predict(X_new))