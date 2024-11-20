#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
import os

matplotlib.use('Agg')

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS = ['←', '↑', '→', '↓']

ACTION_PROB = 0.25
IMAGES_FOLDER = './Images'

# Ensure Images folder exists
os.makedirs(IMAGES_FOLDER, exist_ok=True)

def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward

def draw_image(image, filename):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for (i, j), val in np.ndenumerate(image):
        if [i, j] == A_POS:
            val = f"{val} (A)"
        if [i, j] == A_PRIME_POS:
            val = f"{val} (A')"
        if [i, j] == B_POS:
            val = f"{val} (B)"
        if [i, j] == B_PRIME_POS:
            val = f"{val} (B')"
        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)
    plt.savefig(f"{IMAGES_FOLDER}/{filename}.png")
    plt.close()

def draw_policy(optimal_values, filename):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals = []
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0], next_state[1]])

        best_actions = np.where(next_vals == np.max(next_vals))[0]
        val = ''.join(ACTIONS_FIGS[ba] for ba in best_actions)

        if [i, j] == A_POS:
            val = f"{val} (A)"
        if [i, j] == A_PRIME_POS:
            val = f"{val} (A')"
        if [i, j] == B_POS:
            val = f"{val} (B)"
        if [i, j] == B_PRIME_POS:
            val = f"{val} (B')"
        tb.add_cell(i, j, width, height, text=val, loc='center', facecolor='white')

    for i in range(len(optimal_values)):
        tb.add_cell(i, -1, width, height, text=i + 1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height / 2, text=i + 1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)
    plt.savefig(f"{IMAGES_FOLDER}/{filename}.png")
    plt.close()

def get_epsilon_greedy_policy(value_vector, epsilon):
    num_actions = len(ACTIONS)
    policy = np.ones((WORLD_SIZE, WORLD_SIZE, num_actions)) * (epsilon / num_actions)

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            action_values = []
            for a, action in enumerate(ACTIONS):
                (next_i, next_j), reward = step([i, j], action)
                action_values.append(reward + DISCOUNT * value_vector[next_i, next_j])
            best_action = np.argmax(action_values)
            policy[i, j, best_action] += (1.0 - epsilon)

    return policy

def evaluate_policy(policy, epsilon_label):
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            state = [i, j]
            index_s = np.ravel_multi_index(state, (WORLD_SIZE, WORLD_SIZE))
            for action_idx, action in enumerate(ACTIONS):
                prob = policy[i, j, action_idx]
                next_state, reward = step(state, action)
                index_next = np.ravel_multi_index(next_state, (WORLD_SIZE, WORLD_SIZE))

                A[index_s, index_next] += prob * DISCOUNT
                b[index_s] -= prob * reward

    value_function = np.linalg.solve(A, b).reshape(WORLD_SIZE, WORLD_SIZE)
    draw_image(np.round(value_function, decimals=2), f"GW1_Policy_Value_epsilon_{epsilon_label}")
    draw_policy(value_function, f"GW1_Policy_Policy_epsilon_{epsilon_label}")
    return value_function

if __name__ == '__main__':
    epsilon_values = [0.2, 0.0]
    for epsilon in epsilon_values:
        V = np.zeros((WORLD_SIZE, WORLD_SIZE))
        test_policy = get_epsilon_greedy_policy(V, epsilon)
        evaluate_policy(test_policy, f"{epsilon:.1f}")
