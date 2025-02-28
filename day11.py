from tglearn import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

ls = pd.read_csv("https://github.com/ageron/data/raw/main/lifesat/lifesat.csv")
X = ls[["GDP per capita (USD)"]].values
y = ls[["Life satisfaction"]].values

ls.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23500, 62500, 4, 9])
plt.show()

model = LinearRegression()

model.fit(X, y)

X_new = [[31721.3]]
print(model.predict(X_new))