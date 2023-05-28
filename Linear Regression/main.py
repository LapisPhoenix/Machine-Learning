import pandas as pd
import matplotlib.pyplot as plt


def gradient_decent(m_now, b_now, points, learning_rate: int | float):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        # Calculate Gradient
        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * learning_rate
    b = b_now - b_gradient * learning_rate

    return m, b


def plot(m, b, x, y, color, list_range):
    plt.scatter(x, y, color=color)
    plt.plot(list(range(list_range[0], list_range[1])),
             [m * x + b for x in range(list_range[0], list_range[1])],
             color=color)
    plt.show()


def calcuate_gradients(m, b, learning_rate, epochs):
    for i in range(epochs):
        if i % 50 == 0:
            print(f"Epoch: {i}")
        m, b = gradient_decent(m, b, data, learning_rate)

    return m, b


if __name__ == '__main__':
    data = pd.read_csv(r'example_dataset.csv')
    m = 0
    b = 0
    learning_rate = 0.0001
    epochs = 300  # Iterations
    color = 'black'

    m, b = calcuate_gradients(m, b, learning_rate, epochs)
    print(f"Optimal m, b: ({m}, {b})")

    plot(m, b, data.x, data.y, color, (0, 99))