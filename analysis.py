import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("students.csv")

print("Preview:")
print(data.head())

print("\nStatistics:")
print(data[['G1','G2','G3']].describe())


# 1. Final Grade Distribution

plt.figure()
data['G3'].hist(bins=15)
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade")
plt.ylabel("Number of Students")
plt.show()


# 2. Study Time vs Final Grade

avg_scores = data.groupby('studytime')['G3'].mean()

plt.figure()
avg_scores.plot(kind='bar')
plt.title("Average Final Grade vs Study Time")
plt.xlabel("Study Time Level")
plt.ylabel("Average Final Grade")
plt.show()


# 3. Correlation Heatmap

corr = data[['G1','G2','G3']].corr()

plt.figure()
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.title("Correlation Between Grades")

plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()

# 4. Scatter Plot

plt.figure()
plt.scatter(data['G1'], data['G3'])

plt.title("G1 vs Final Grade")
plt.xlabel("G1")
plt.ylabel("G3")

plt.show()


# 5. MACHINE LEARNING MODEL

X = data[['G1','G2']]
y = data['G3']

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("\nSample Predictions:")
for i in range(5):
    print(f"Actual: {y.iloc[i]}, Predicted: {round(predictions[i],2)}")


# 6. Actual vs Predicted Plot

plt.figure()
plt.scatter(y, predictions)

plt.xlabel("Actual Final Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Comparison")

plt.show()
