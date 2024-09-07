import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import time

# 1a
path = "Assignment 1\SpotifyFeatures.csv"
df = pd.read_csv(path, header=0)
print(f"Number of samples (songs): {df.shape[0]}")
print(f"Number of features (song properties): {df.shape[1]}")

# 1b
pop_samples = len(df[df['genre'] == 'Pop'])
classical_samples = len(df[df['genre'] == 'Classical'])
print(pop_samples, "samples belongs to pop genre")
print(classical_samples, "samples belongs to classical genre")

filtered_df = df[(df['genre'] == 'Pop') | (df['genre'] == 'Classical')]
conditions = [
    (filtered_df['genre'] == 'Pop'), 
    (filtered_df['genre'] == 'Classical')
]
choices = [1, 0]  # 1 for pop, 0 for classical

filtered_df['label'] = np.select(conditions, choices)

parameters = ['liveness', 'loudness']

# 1c
# Ensure the split maintains the same class distribution.

array_liveness_loudness = filtered_df[parameters].copy()
vector_label = filtered_df['label']
#normalize the data using min max normalization
array_liveness_loudness['loudness'] = (array_liveness_loudness['loudness'] - array_liveness_loudness['loudness'].min()) / (array_liveness_loudness['loudness'].max() - array_liveness_loudness['loudness'].min())
# Split the data
# X_train_data, X_test_data, y_train_labels, y_test_labels = train_test_split(array_liveness_loudness, vector_label, test_size=0.2, random_state=28)
stratifiedsplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=28)
# stratifiedsplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
for trainindex, testindex in stratifiedsplit.split(array_liveness_loudness, vector_label):
    X_train_data, X_test_data = array_liveness_loudness.iloc[trainindex], array_liveness_loudness.iloc[testindex]
    y_train_labels, y_test_labels = vector_label.iloc[trainindex], vector_label.iloc[testindex]
# 1d Plot the samples on the liveness vs loudness plane, with a different color for each class. From the plot, will the classification be an easy task? why?
# Plotting the samples on the liveness vs loudness plane, with a different color for each class
plt.figure(figsize=(10, 6))

# Plot with distinct colors for each class
scatter = plt.scatter(X_train_data['liveness'], X_train_data['loudness'], c=y_train_labels, cmap='viridis')

# Adding labels
plt.xlabel('Liveness', fontsize=12)
plt.ylabel('Loudness', fontsize=12)
plt.title('Normalized Liveness vs Loudness', fontsize=15)

# Create the legend of classical and pop songs
plt.legend(['Classical', 'Pop'])
# Show plot
plt.show()

# Problem 2

def stochastic_gradient_descent(X_train_data, y_train_labels, X_test_data, y_test_labels, learning_rate=0.01, maximum_epochs=1000, batch_size=32, tolerance=1e-3):
    number_of_samples, number_of_features = X_train_data.shape
    # initialize the weights and bias as zeros (or alternatively random values)
    weights = np.zeros(number_of_features)
    bias = 0
    training_errors = []
    validation_errors = []
    for epoch in range(maximum_epochs):
        # shuffle the data to prevent the model from learning the order of the data
        shuffled_index = np.random.permutation(number_of_samples) 
        X_train_shuffled = X_train_data[shuffled_index]
        y_labels_shuffled = y_train_labels[shuffled_index]
        epoch_error_sum = 0
        # iterate over the data in batches for every ⟨x[i],y[i]⟩∈D:
        for start in range(0, number_of_samples, batch_size):
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_labels_shuffled[start:end]
            # compute the prediction for the current batch
            z = np.dot(X_batch, weights) + bias
            y_prediction = sigmoid(z)
            error = y_prediction - y_batch
            epoch_error_sum += np.sum(error**2)
            # compute the gradient of the loss with respect to the weights and bias
            gradient_weights = np.dot(X_batch.T, error) / X_batch.shape[0] 
            gradient_bias = np.mean(error)
            
            # update parameters w:=w+Δw,b:=+Δb
            weights = weights - learning_rate * gradient_weights
            bias = bias - learning_rate * gradient_bias
        epoch_error = epoch_error_sum / number_of_samples        
        training_errors.append(epoch_error)
        
        # Compute the validation error
        z_val = np.dot(X_test_data, weights) + bias
        val_predictions = sigmoid(z_val)
        val_error = val_predictions - y_test_labels
        val_error_sum = np.sum(val_error ** 2)
        val_error_mean = val_error_sum / X_test_data.shape[0]
        validation_errors.append(val_error_mean)        
        
        # check for convergence
        if np.linalg.norm(gradient_weights) < tolerance:
            break
    return weights, bias, training_errors, validation_errors
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def solve_for_learning_rate_and_create_subplot(X_train_data, y_train_labels, X_test_data, y_test_labels, learning_rates):
    fig, axes = plt.subplots(len(learning_rates), 4, figsize=(20, 15))

    for i, lr in enumerate(learning_rates):
        # measure the time taken to train the model
        time_start = time.time()
        weights, bias, training_errors, validation_errors = stochastic_gradient_descent(X_train_data.values, y_train_labels.values, X_test_data=X_test_data.values, y_test_labels=y_test_labels.values, learning_rate=lr, maximum_epochs=1000, batch_size=32, tolerance=1e-3)

        y_pred_train = sigmoid(np.dot(X_train_data.values, weights) + bias)
        y_pred_test = sigmoid(np.dot(X_test_data.values, weights) + bias)
        y_pred_train = np.where(y_pred_train > 0.5, 1, 0)  # convert the prediction to 0 or 1
        y_pred_test = np.where(y_pred_test > 0.5, 1, 0)  # convert the prediction to 0 or 1

        train_accuracy = accuracy_score(y_train_labels, y_pred_train)
        test_accuracy = accuracy_score(y_test_labels, y_pred_test)

        confusion = confusion_matrix(y_test_labels, y_pred_test)
        time_end = time.time()
        time_taken = time_end - time_start

        print(f"Confusion matrix for learning rate {lr} is: \n{confusion}\nTrain accuracy: {train_accuracy}\nTest accuracy: {test_accuracy}\nTime taken is {round(time_taken, 2)} seconds.\n-----------------")

        # Suggesting Classical songs
        suggested_classical_songs = filtered_df[(filtered_df['label'] == 0)].sample(1) # Random suggestion
        print("Suggested Classical song for Pop fans: \n", suggested_classical_songs[['track_name', 'artist_name']])



        # Plotting the decision boundary
        axes[i, 0].scatter(X_train_data['liveness'], X_train_data['loudness'], c=y_train_labels, cmap='viridis')
        axes[i, 0].set_xlabel('Liveness', fontsize=12)
        axes[i, 0].set_ylabel('Loudness', fontsize=12)
        axes[i, 0].set_title(f'Normalized Liveness vs Loudness (LR={lr})', fontsize=12)
        x = np.linspace(0, 1, 100)
        y = -(weights[0] * x + bias) / weights[1]
        axes[i, 0].plot(x, y, '-r', label='Decision boundary')
        axes[i, 0].legend(loc='upper left')

        # Plotting the training error vs epochs
        axes[i, 1].plot(training_errors, label=f'Learning rate {lr}')
        axes[i, 1].set_xlabel('Epochs')
        axes[i, 1].set_ylabel('Training Error')
        axes[i, 1].set_title(f'Training Error vs Epochs (LR={lr})', fontsize=12)
        axes[i, 1].legend()

        # Plotting the validation error vs epochs
        axes[i, 2].plot(validation_errors, label=f'Learning rate {lr}')
        axes[i, 2].set_xlabel('Epochs')
        axes[i, 2].set_ylabel('Validation Error')
        axes[i, 2].set_title(f'Validation Error vs Epochs (LR={lr})', fontsize=12)
        axes[i, 2].legend()

        # Plotting the confusion matrix
        cmd = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[0, 1])
        cmd.plot(ax=axes[i, 3], values_format='d', cmap='viridis')
        axes[i, 3].set_title(f'Confusion Matrix (LR={lr})')

    plt.tight_layout()
    plt.show()
# learning_rates = [0.1, 0.01, 0.001]
learning_rates = [0.1]
solve_for_learning_rate_and_create_subplot(X_train_data, y_train_labels, X_test_data, y_test_labels, learning_rates)

