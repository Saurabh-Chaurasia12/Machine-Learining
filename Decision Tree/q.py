# Data Loading and Preprocessing:
# The dataset is loaded from a URL using pd.read_csv.
# The data is reshaped and split into features and target.

# Train-Test Split:
# The data is split into training and testing sets using train_test_split, with 80% of the data used for training and 20% for testing.

# Model Training and Evaluation:
# A loop iterates over a range of depths (1 to 20) to train a DecisionTreeRegressor with different maximum depths.
# The model is trained on the training data (X_train, y_train).
# Predictions are made on the training data, and the RMSE is calculated to evaluate the model's performance.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data1 = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
data = raw_df.values[1::2, 2]
X_train, X_test, y_train, y_test = train_test_split(data1, data, test_size=0.2, random_state=42)


rmse_values = []
train_rmse_values = []
depths = range(1, 21)
plt.figure(figsize=(15, 10))

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_rmse_values.append(train_rmse)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)
    plt.subplot(4, 5, depth)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Original')
    plt.ylabel('Predicted')
    plt.title(f'Depth: {depth}, RMSE: {rmse:.2f}')

plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(depths, rmse_values, marker='o',label='Test RMSE')
plt.plot(depths, train_rmse_values, marker='o',label='Train RMSE')
plt.xlabel('Depth of the tree')
plt.ylabel('RMSE')
plt.title('RMSE vs Depth of the tree')
plt.legend()
plt.grid(True)
plt.show()