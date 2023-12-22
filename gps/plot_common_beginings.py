import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def perform_linear_regression(file_path, label, color):
    # Load data from CSV file
    data = pd.read_csv(file_path)

    # Extract latitude (X) and longitude (Y) from the dataset
    latitude = data.iloc[:, 1].values.reshape(-1, 1)  # Assuming the second column is latitude
    longitude = data.iloc[:, 2].values.reshape(-1, 1)  # Assuming the third column is longitude

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(latitude, longitude)

    # Display the slope (coefficient) and intercept
    print(f'{label}: Slope: {model.coef_[0]}, Intercept: {model.intercept_}')

    # Plot the original data and the linear regression line
    plt.scatter(latitude, longitude, label=f'{label} Data', color=color)
    plt.plot(latitude, model.predict(latitude), label=f'{label} Linear Regression', color='black', linestyle='--')

    # Display starting point of the linear regression
    linear_reg_x0 = latitude[0][0]
    linear_reg_y0 = model.predict(latitude)[0][0]
    plt.scatter(linear_reg_x0, linear_reg_y0, marker='x', color='red', label='Starting Point', s=100)

    x_point = linear_reg_x0
    y_point = linear_reg_y0
    slope = -1 / model.coef_[0]
    x_values = latitude  # Adjust the range as needed
    y_values = slope * (x_values - x_point) + y_point
    plt.plot(x_values, y_values, label='Line with Slope -1/6 through (3349.558668, -2214.2078155709423)', color='red')
    # plt.scatter(x_point, y_point, marker='o', color='red', label='Given Point')


# Replace these file paths with the actual file paths
file_paths = [('lane1.csv', 'Lane 1', 'blue'), ('lane2.csv', 'Lane 2', 'green'), ('lane3.csv', 'Lane 3', 'red')]

# Plot all datasets on the same graph
for file_path, label, color in file_paths:
    perform_linear_regression(file_path, label, color)
    break

# Add labels and a legend
plt.xlabel('Latitude')
plt.ylabel('Longitude')
# plt.legend()

# Show the plot
plt.show()
