import matplotlib.pyplot as plt
import numpy as np


def grayscale(screen):
    screen_ = screen.flatten()
    screen_gray = np.zeros(96*96)

    batch_index = np.arange(96*96)
    rgb_index = np.arange(0, 96*96*3, 3)

    screen_gray[batch_index] = to_grey(screen_[rgb_index], screen_[rgb_index+1], screen_[rgb_index+2])
    screen_gray = np.reshape(screen_gray, (96, 96, 1))

    return screen_gray


def to_grey(red, green, blue):
    return (0.3*red + 0.59*green + 0.11*blue) / 255.0


def plot_learning(scores, avg_scores, title, filename):
    size = len(scores)
    x = [i+1 for i in range(size)]

    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.plot(x, scores, color='orange', label='score (orginal)')
    plt.plot(x, avg_scores, color='red', label='score (100 moving averages)')

    plt.title(title)
    plt.legend()

    plt.savefig(filename)






