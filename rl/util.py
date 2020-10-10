import numpy as np


def get_action(state, num_actions, model, epsilon):
    # Exploration by epsilon
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    # Exploitation by greedy
    else:
        return np.argmax(model(state))
