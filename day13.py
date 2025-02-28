import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


mpg = sns.load_dataset('mpg') 
# 데이터 전처리
mpg.drop(['name'], axis=1, inplace=True)  
mpg.dropna(inplace=True)  
mpg = pd.get_dummies(mpg, columns=['origin'], drop_first=True)  

X = mpg.drop(['mpg'], axis=1)
y = mpg['mpg']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 선택 및 학습
model = LinearRegression() 
model.fit(X_train, y_train)

# 테스트 (예측 값 생성)
y_pred = model.predict(X_test)

# 성능 측정
mse = mean_squared_error(y_test, y_pred)  
r2 = r2_score(y_test, y_pred)  

# 결과 출력
print(f"Mean Squared Error : {mse}")
print(f"R2 Score : {r2}")

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Actual MPG vs Predicted MPG')
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.grid(True)
plt.show()