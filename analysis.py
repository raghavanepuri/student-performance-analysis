import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("students.csv")

# Display basic information
print("First few rows of dataset:")
print(data.head())

print("\nSummary statistics:")
print(data[['G1', 'G2', 'G3']].describe())


# Plot 1: Final grade distribution

plt.figure()
data['G3'].hist(bins=15)
plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.show()


# Plot 2: Study time vs performance

avg_scores = data.groupby('studytime')['G3'].mean()

plt.figure()
avg_scores.plot(kind='bar')
plt.title("Average Final Grade vs Study Time")
plt.xlabel("Study Time Level (1-4)")
plt.ylabel("Average Final Grade")
plt.show()


# Plot 3: Correlation heatmap

corr_matrix = data[['G1', 'G2', 'G3']].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

plt.figure()
plt.imshow(corr_matrix, cmap='coolwarm')
plt.colorbar()
plt.title("Correlation Between Grades")

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

plt.show()


# Plot 4: G1 vs G3 relationship

plt.figure()
plt.scatter(data['G1'], data['G3'])

plt.title("Relationship Between First and Final Grades")
plt.xlabel("G1 (First Period Grade)")
plt.ylabel("G3 (Final Grade)")

plt.show()