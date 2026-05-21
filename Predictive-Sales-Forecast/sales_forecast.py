# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("data/sales_data.csv")

# Display dataset
print("Sales Dataset:\n")
print(data)

# Convert months into numbers
data['Month_Number'] = range(1, len(data) + 1)

# Prepare data
X = data[['Month_Number']]
y = data['Sales']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict next month sales
next_month = [[13]]
prediction = model.predict(next_month)

print("\nPredicted Sales for Next Month:", int(prediction[0]))

# Plot graph
plt.figure(figsize=(8,5))
plt.plot(data['Month'], data['Sales'], marker='o')

# Graph labels
plt.title("Sales Forecast Analysis")
plt.xlabel("Months")
plt.ylabel("Sales")

# Save chart
plt.savefig("images/sales_chart.png")

# Show graph
plt.show()

print("\nChart saved in images folder!")