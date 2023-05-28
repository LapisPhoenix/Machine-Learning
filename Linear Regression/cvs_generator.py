import os
import csv
import random


def generate_data_points(num_points: int, slope: int | float, intercept: int | float, outlier_prob: int | float):
    data_points: list = []
    for i in range(num_points):
        x: int = i
        # Generate outliers with a certain probability
        if random.random() < outlier_prob:
            # Generate outlier values dispersed from low to high, somewhat following the trend
            y: float = slope * x + intercept + random.uniform(-5, 5) + random.uniform(-50, 50)
        else:
            y: float = slope * x + intercept + random.uniform(-5, 5)  # Add some random noise
        data_points.append((x, y))
    return data_points


def save_data_points_to_csv(data_points: list, csv_file_path: str | os.PathLike):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y'])  # Write header
        writer.writerows(data_points)  # Write data points
    print(f"Data points saved to \"{csv_file_path}\".")


def find_lowest_and_highest_x(data_points: list):
    lowest_x = float('inf')  # Initialize with positive infinity
    highest_x = float('-inf')  # Initialize with negative infinity

    for x, _ in data_points:
        if x < lowest_x:
            lowest_x = x
        if x > highest_x:
            highest_x = x

    return lowest_x, highest_x


if __name__ == '__main__':
    # Example usage
    num_points = 100
    slope = 2.5
    intercept = 1.5
    outlier_prob = 0.8  # Probability of generating an outlier, higher = more outliers
    csv_file_path = "example_dataset.csv"

    data_points = generate_data_points(num_points, slope, intercept, outlier_prob)
    save_data_points_to_csv(data_points, csv_file_path)

    lowest_x, highest_x = find_lowest_and_highest_x(data_points)
    print("Lowest x value:", lowest_x)
    print("Highest x value:", highest_x)
