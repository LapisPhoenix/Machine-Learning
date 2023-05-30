import csv
import os
import pickle
import math
import pandas as pd
import numpy as np
# import matplotlib.pyplot as pyplot
from sklearn.model_selection import train_test_split    # Splitting Data
from sklearn.preprocessing import StandardScaler    # No Bias, turn everything on a scale of -1, 1
from sklearn.neighbors import KNeighborsClassifier  # The KNN
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score  # Testing
# from matplotlib import style


# AI Functions
def clean_data(data, save: bool = False, save_path: str = 'cleaned_data.csv'):
    """Clean the data provided, getting rid of all zeros, and replaces with a guess."""
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

    for column in zero_not_accepted:
        data[column] = data[column].replace(0, np.NaN)
        mean = int(data[column].mean(skipna=True))
        data[column] = data[column].replace(np.NaN, mean)

    if save:
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data.columns)
            writer.writerows(data.values)

    return data


def load(file: str = r'diabetes.csv', clean: bool = True, save_to_file: bool = False, save_path: str = None):
    """Load the data"""
    dataset = pd.read_csv(file)

    if clean and save_to_file and save_path is not None:
        dataset = clean_data(data=dataset, save=True, save_path=save_path)
    elif clean and save_to_file:
        dataset = clean_data(data=dataset, save=True)
    elif clean:
        dataset = clean_data(dataset)

    return dataset


def display_dataset(data, cm, score, accurate_score):
    """Display the data in a nice format."""
    data_length = len(data)
    headers = data.head()

    print(f'Dataset Length: {data_length}\nData:\n{headers}')
    print(f"cm:\n{cm}")
    print(f"score: {score} ({round(score, 2) * 100}%)")
    print(f"accuracy score: {accurate_score} ({round(accurate_score, 2) * 100}%)")


def split_data(data, test_size=0.2):
    """Split the data up into training chunks, takes by default 20% of data provided."""
    x = data.iloc[:, 0:8]   # : = All Rows, only at columns 0 to 8, which is actually 0 to 7
    y = data.iloc[:, 8]     # : = All Rows, only column 7

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=test_size) # random_state = seed
    # 0.2 = 20% of the data will be set aside for training.

    return x_train, x_test, y_train, y_test


def fit_data(x_train, x_test):
    """Change data from -INF to INF to a value between -1, 1."""
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)  # Is not apart of the train

    return x_train, x_test


def find_k(y_test):
    """Tries to find the best possible K value."""
    possible_k = math.sqrt(len(y_test))

    if int(possible_k) % 2 == 0:
        return int(possible_k - 1)
    else:
        return int(possible_k)


def fitness(x_train, y_train, k):
    """Create, and train the model."""
    classifier = KNeighborsClassifier(n_neighbors=k, p=1, metric='euclidean')
    classifier.fit(x_train, y_train)

    return classifier


def predict(model, x_test):
    """Predict the model"""
    y_pred = model.predict(x_test)

    return y_pred


def evaluate_model(model, x_test, y_test):
    """Evaluate the model, with f1 score and accuracy score + matrix"""
    prediction = predict(model, x_test)

    cm = confusion_matrix(y_test, prediction)
    score = f1_score(y_test, prediction)
    accurate_score = accuracy_score(y_test, prediction)

    return cm, score, accurate_score


def scalar(data):
    """Split the data into smaller chunks, and test it."""
    x_train, x_test, y_train, y_test = split_data(data)

    x_train, x_test = fit_data(x_train, x_test)

    return x_train, x_test, y_train, y_test


def find_best(data, iterations):
    """Find the best model with several iterations."""
    best = 0.0

    for _ in range(iterations):
        x_train, x_test, y_train, y_test = scalar(data)
        k = find_k(y_test)
        model = fitness(x_train, y_train, k)

        cm, score, accurate_score = evaluate_model(model, x_test, y_test)

        if accurate_score > best:
            best = accurate_score
            save_model(model)

    return model, best


def save_model(model):
    """Save the KNN Model"""
    with open('knnmodel.pickle', 'wb') as file:
        pickle.dump(model, file)


def load_model(model_file):
    """Load the KNN Model"""
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    return model


# Util
def check_response(response: str):
    """Check if a response is yes or no."""
    response = response.casefold()

    if response in ['yes', 'y']:
        return True
    elif response in ['no', 'n']:
        return False


def model_exists(file):
    """Check if a saved model already exists. """
    saved_model = False
    for file in os.listdir("./"):
        if file == "knnmodel.pickle":
            saved_model = True
            break
        else:
            continue

    return saved_model


def main():
    # Load the data, and clean it
    data = load(file=r'cleaned_data.csv', clean=False)
    x_train, x_test, y_train, y_test = scalar(data)

    model_exists_ = model_exists('knnmodel.pickle')

    # Check if we already have a model
    if not model_exists_:
        # Find the best k value

        k = find_k(y_test)

        # Create the model

        model = fitness(x_train, y_train, k)
    else:
        # Load the model
        model = load_model("knnmodel.pickle")

    cm, score, accurate_score = evaluate_model(model, x_test, y_test)

    display_dataset(data, cm, score, accurate_score)

    if not model_exists_:
        print(f"Saving Best Possible Model.")
        find_best(data, 30)
        print(f"Saved!")


if __name__ == '__main__':
    main()
