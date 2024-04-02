import datetime as dt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import data science tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
import seaborn as sns


# We define the class of a simple Neural Network through the use of the PyTorch library
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        #define the first layer which has neurons = <input_size> with edges per neuron = <hidden_size>
        self.fc0 = nn.Linear(input_size, hidden_size)
        #defines the hidden layers with <hidden_size> neurons and <hidden_size> outgoing edges
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        # defines the second hidden layer
        #self.fc2 = nn.Linear(hidden_size, hidden_size)
        # defines the third hidden layer
        #self.fc3 = nn.Linear(hidden_size, hidden_size)
        #defines the output layer with hidden_size connections going to output_size neurons
        self.fcf = nn.Linear(hidden_size, output_size)
        #defines the relu function used as the activation function between neurons
        self.relu = nn.ReLU()
        #defines the final function used on the forward pass
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(x)

        x = self.fc1(x)
        x = self.relu(x)

        #x = self.fc2(x) # Second hidden layer
        # x = self.relu(x)

        # x = self.fc3(x) # Third hidden layer
        # x = self.relu(x)

        x = self.fcf(x)
        x = self.sigmoid(x)
        #x = self.softmax(x)
        #x = x.squeeze(1) # added to remove additional dimension [400, 1] added during pytorch linear layering removed for softmax
        return x

        # # Added a second layer with lines
        # self.fc2a = nn.Linear(hidden_size, hidden_size)
        # x = self.relu(x)
        # x = self.fc2a(x)

class MoonShot2:
    def __init__(self, buytable: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        self.buy_trades_core = buytable
        self.buy_trades_core.fillna(0, inplace=True)
        self.preprocess_data(test_size, random_state)
        self.train_and_tune()
        self.evaluate()
        self.reshape()
        self.train_model()

    def preprocess_data(self, test_size: float = 0.2, random_state: int = 42):
        # We start by changing the string values for the profitable column into their boolean equivalents.
        # This column will become out target (y) values
        self.buy_trades_core["Profitable"] = self.buy_trades_core["Profitable"].replace("Yes", 1)
        self.buy_trades_core["Profitable"] = self.buy_trades_core["Profitable"].replace("No", 0)

        # Removing columns that are unused for future subsets.
        # Columns are removed upstream to avoid corrupting future scaling, normalization, or correlations with bad data.
        self.buy_trades_core = self.buy_trades_core.drop(columns= ["Buy vol"])
        buy_trades = self.buy_trades_core

        # Two new dataframes are created, one contains all of the input features of the dataset (x), the other contains the target values (y)
        buy_x = buy_trades.drop(columns= "Profitable")
        buy_y = buy_trades["Profitable"]

        # The input features are then preprocessed using standard scaling and normalization techniques.
        # Scaling helps prevent feature domination in model training and increases convergence in the gradient descent used in optimization functions
        # The scaler is initialized from the scikit learn library and then fit to the features of our dataset
        scaler = StandardScaler()
        scaler.fit(buy_x)

        # We finalize the process by applying the scaler to the data in our dataframe. This is stored as a numpy array.
        # buy_x_scaled = scaler.transform(buy_x)

        # The dataset is split into the training and test sets.
        # Data is shuffled to prevent overfitting to subsets and reduce underlying patterns in time based data.
        # We use the industry standard of starting with an 80/20 split on the data set, adjusting if needed based on task complexity and set size
        self.buy_x_train, self.buy_x_val, self.buy_y_train, self.buy_y_val = train_test_split(buy_x, buy_y, test_size=0.2, random_state = 42)
        # self.buy_x_train, self.buy_x_test, self.buy_y_train, self.buy_y_test = train_test_split(buy_x_scaled, buy_y, test_size=0.2, random_state = 42)

        # Apply the scaler to the training and validation data separately to prevent data leakage
        self.buy_x_train = scaler.transform(self.buy_x_train)
        self.buy_x_val = scaler.transform(self.buy_x_val)


        # self.buy_x_train.shape

        # # We convert the numpy arrays into PyTorch tensors to be used in the neural network
        # try:
        #     self.buy_y_train_tensor = torch.from_numpy(self.buy_y_train.values)
        #     self.buy_y_train_tensor = self.buy_y_train_tensor.float()
        #     print("Created Y train tensor")
        #     self.buy_y_validation_tensor = torch.from_numpy(self.buy_y_test.values)
        #     self.buy_y_validation_tensor = self.buy_y_validation_tensor.float()
        #     print("Created Y validation tensor")
        # except ValueError:
        #     print("Error: buy_y_train or buy_y_test contains non-convertible values.")
        # try:
        #     self.buy_x_train_tensor = torch.from_numpy(self.buy_x_train)
        #     self.buy_x_train_tensor = self.buy_x_train_tensor.float()
        #     print("Created X train tensor")
        #     print("Createc X validation tensor")
        #     self.buy_x_validation_tensor = torch.from_numpy(self.buy_x_test)
        #     self.buy_x_validation_tensor = self.buy_x_validation_tensor.float()
        # except ValueError:
        #     print("Error: buy_x_train contains non-convertible values.")

    def train_and_tune(self):
        # We define the hyperparameters of the neural network
        input_size = self.buy_x_train.shape[1]
        hidden_size = 56
        # hidden_size = 64
        output_size = 1
        # learning_rate = 0.00001
        learning_rate = 0.001
        # self.num_epochs = 100
        self.num_epochs = 1000

        # Loss: 0.4661
        # Loss with 2 layers: 0.4745


        self.moonShot_buy = SimpleNN(input_size, hidden_size, output_size)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.moonShot_buy.parameters(), lr=learning_rate)
        # self.optimizer = torch.optim.SGD(self.moonShot_buy.parameters(), lr=learning_rate, momentum=0.9)
        # self.optimizer = torch.optim.Adagrad(self.moonShot_buy.parameters(), lr=learning_rate)

        # # We define the neural network and the optimizer
        # self.model = SimpleNN(input_size, hidden_size, output_size)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)


        # Initialize your loss function here
        self.criterion = nn.BCELoss()

        # Convert your data to tensors
        self.buy_x_train_tensor = torch.from_numpy(self.buy_x_train).float()
        self.buy_y_train_tensor = torch.from_numpy(self.buy_y_train.values).float()
        self.buy_x_validation_tensor = torch.from_numpy(self.buy_x_val).float()
        self.buy_y_validation_tensor = torch.from_numpy(self.buy_y_val.values).float()

        # Reshape target tensor to match output shape
        self.buy_y_train_tensor = self.buy_y_train_tensor.view(-1, 1)
        self.buy_y_validation_tensor = self.buy_y_validation_tensor.view(-1, 1)


    def evaluate(self):
        """
        This function evaluates the model performance on a validation set.

        Args:
            None

        Returns:
            val_loss: The validation loss (calculated using the criterion function).
            val_accuracy: The validation accuracy.
        """
        with torch.no_grad():  # Deactivate gradient calculation for validation
            # Forward pass on validation set
            self.val_outputs = self.moonShot_buy(self.buy_x_validation_tensor)

            # Assuming self.val_outputs is the input tensor and self.buy_y_validation_tensor is the target tensor
            self.val_outputs = self.val_outputs.view(self.buy_y_validation_tensor.shape)

            # Calculate loss
            self.val_loss = self.criterion(self.val_outputs, self.buy_y_validation_tensor)

            # Calculate accuracy
            self.predicted = (self.val_outputs > 0.5).float()  # Thresholding for binary classification
            self.val_accuracy = (self.predicted == self.buy_y_validation_tensor).sum() / len(self.buy_y_validation_tensor)

        return self.val_loss.item(), self.val_accuracy.item()

    def reshape(self):
        # Reshape the data to fit the neural network
        # self.buy_x_train_tensor = self.buy_x_train_tensor.view(-1, len(self.buy_x_train_tensor[0]))
        # self.buy_y_train_tensor = self.buy_y_train_tensor.view(-1, 1)

        # self.buy_x_validation_tensor = self.buy_x_validation_tensor.view(-1, len(self.buy_x_validation_tensor[0]))
        # self.buy_y_validation_tensor = self.buy_y_validation_tensor.view(-1, 1)

        # Reshape target tensor to match output shape
        self.buy_y_train_tensor = self.buy_y_train_tensor.view(-1, 1)
        self.buy_y_validation_tensor = self.buy_y_validation_tensor.view(-1, 1)

    def train_model(self):
        # Implement early stopping
        best_val_loss = float('inf')
        patience = 10  # Number of epochs with no improvement after which training will be stopped
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.moonShot_buy.train()
            self.optimizer.zero_grad()

            # The forward pass as defined in the neural network architecture
            self.outputs = self.moonShot_buy(self.buy_x_train_tensor)
            self.loss = self.criterion(self.outputs, self.buy_y_train_tensor)

            # Backward pass of the calculated loss
            self.loss.backward()
            self.optimizer.step()

            # Evaluate on validation set
            self.val_loss, self.val_accuracy = self.evaluate()

            # if self.val_loss < best_val_loss:
            #     best_val_loss = self.val_loss
            #     patience_counter = 0
            # else:
            #     patience_counter += 1

            # if patience_counter >= patience:
            #     print('Early stopping')
            #     break

            if (epoch + 1) % 2 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {self.loss.item():.4f}, Val Loss: {self.val_loss:.4f}, Val Accuracy: {self.val_accuracy:.4f}')
