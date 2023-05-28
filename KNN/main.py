import csv
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split    # Splitting Data
from sklearn.preprocessing import StandardScaler    # No Bias, turn everything on a scale of -1, 1
from sklearn.neighbors import KNeighborsClassifier  # The KNN
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score  # Testing


def clean_data(data, save: bool = False, save_path: str = 'cleaned_data.csv'):
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
    dataset = pd.read_csv(file)

    if clean and save_to_file and save_path is not None:
        dataset = clean_data(data=dataset, save=True, save_path=save_path)
    elif clean and save_to_file:
        dataset = clean_data(data=dataset, save=True)
    elif clean:
        dataset = clean_data(dataset)

    return dataset


def display_dataset(data):
    data_length = len(data)
    headers = data.head()

    print(f'Dataset Length: {data_length}\nData:\n{headers}')


def check_response(response: str):
    response = response.casefold()

    if response in ['yes', 'y']:
        return True
    elif response in ['no', 'n']:
        return False


def split_data(data):
    x = data.iloc[:, 0:8]   # : = All Rows, only at columns 0 to 8, which is actually 0 to 7
    y = data.iloc[:, 8]     # : = All Rows, only column 7

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)    # random_state = seed
    # 0.2 = 20% of the data will be set aside for training.

    return x_train, x_test, y_train, y_test

def fit_data(x_train, x_test):
    # Change data from 0.. INF to a value between -1, 1
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test) # Is not apart of the train

    return x_train, x_test


def find_k(y_test):
    possible_k = math.sqrt(len(y_test))

    if int(possible_k) % 2 == 0:
        return int(possible_k - 1)
    else:
        return int(possible_k)


def fitness(x_train, y_train, x_test, k):
    classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    classifier.fit(x_train, y_train)

    return classifier


def predict(model, x_test):
    y_pred = model.predict(x_test)

    return y_pred


def evaluate_model(model, x_test, y_test):
    prediction = predict(model, x_test)

    cm = confusion_matrix(y_test, prediction)
    score = f1_score(y_test, prediction)
    accurate_score = accuracy_score(y_test, prediction)

    return cm, score, accurate_score


if __name__ == '__main__':
    # Load the data, and clean it
    data = load(file='cleaned_data.csv', clean=False)

    # Now that our data is cleaned, we can scale it.

    x_train, x_test, y_train, y_test = split_data(data)

    x_train, x_test = fit_data(x_train, x_test)

    # Find the best k value

    k = find_k(y_test)

    model = fitness(x_train, y_train, x_test, k)

    cm, score, accurate_score = evaluate_model(model, x_test, y_test)

    print(f"cm: {cm}\nscore: {score}\naccuracy score: {accurate_score}")
