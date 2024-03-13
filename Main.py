
import numpy as np
import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *

TEST_SIZE = 0.3
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features, k):

        # array to hold predicted lables
        predictions = []

        # calculate thr Euclidean distance between each feature and the set of training features. 
        for feature in features:
            distances = np.sqrt(np.sum(np.square(np.subtract(self.trainingFeatures, feature)), axis=1))
            nearest_indices = np.argsort(distances)[:k]
            nearest_labels = [self.trainingLabels[i] for i in nearest_indices]
            prediction = np.argmax(np.bincount(nearest_labels))
            predictions.append(prediction)
        return predictions
    
        """
        Given a list of features vectors of testing examples
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors
        """


def load_data(filename):

    features_vectors = []
    traget_labels = []

    with open(filename, 'r') as file:

        csv_reader = csv.reader(file)
        for row in csv_reader:

            #row[:-1] is used to slice the row list, excluding the last element. This creates a new list of feature values
            feature_values = [float(value) for value in row[:-1]]

            #row[-1] is used to access the last element of each row in the CSV file, which represents  the label associated with the features.
            label = int(row[-1]) 

            features_vectors.append(feature_values)
            traget_labels.append(label)
    return features_vectors, traget_labels

    """
    Load spam data from a CSV file `filename` and convert into a list of
    features vectors and a list of target labels. Return a tuple (features, labels).

    features vectors should be a list of lists, where each list contains the
    57 features vectors

    labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """


def preprocess(features):

    features = np.array(features)

    feature_mean = np.mean(features, axis=0)
    feature_standard = np.std(features, axis=0)
    preprocessed_features = (features - feature_mean) / feature_standard

    return preprocessed_features.tolist()

    """
    normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation
    """


def train_mlp_model(features, labels):

    model = MLPClassifier(hidden_layer_sizes=(10, 5),  activation='logistic' , max_iter=1500,random_state=1)
    #sigmoid activation function
    #edit iterations
    model.fit(features, labels)
    return model

    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    first layer with 10 neurons, and the second layer with 5 neurons
    """

def evaluate(labels, predictions):

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return accuracy, precision, recall, f1

    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("\n**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Generate confusion matrix
    cm_knn = confusion_matrix(y_test, predictions)



    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    mlp_predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, mlp_predictions)

    # Print results
    print("\n**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Generate confusion matrix
    cm_mlp = confusion_matrix(y_test, mlp_predictions)

    print("\nConfusion Matrix - k-NN:")
    print(cm_knn)

    print("\nConfusion Matrix - MLP:")
    print(cm_mlp)



if __name__ == "__main__":
    main()
