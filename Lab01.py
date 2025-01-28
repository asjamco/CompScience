import matplotlib.pyplot as plt

def perform_linear_regression(x_values, y_values, input_x):
    # Number of data points
    data_count = len(x_values)
    
    # Calculate necessary summations
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xy = sum(x_values[i] * y_values[i] for i in range(data_count))
    sum_x_squared = sum(val ** 2 for val in x_values)
    
    # Calculate slope (m) and intercept (c)
    slope = (data_count * sum_xy - sum_x * sum_y) / (data_count * sum_x_squared - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / data_count
    
    # Predict the y value for the given x
    predicted_y = intercept + slope * input_x
    
    # Plot data points
    plt.scatter(x_values, y_values, color="red", label="Observed Data")
    plt.scatter(input_x, predicted_y, color="yellow", label="Prediction")

    # Plot the regression line
    regression_y_values = [intercept + slope * x for x in x_values]
    plt.plot(x_values, regression_y_values, color="blue", label="Regression Line")
    
    # Add labels, title, and legend
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.title("Linear Regression Visualization")
    plt.legend()
    plt.show()
    
    return predicted_y

# Example data
x_data = [140, 155, 159, 179, 192, 200, 212]
y_data = [60, 62, 67, 70, 71, 72, 75]
input_x_value = 192

# Execute the function
predicted_value = perform_linear_regression(x_data, y_data, input_x_value)
print(f"Predicted y for x = {input_x_value} is: {predicted_value}")
