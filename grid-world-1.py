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

#left, up, right, down
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

def figure_3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # bellman equation
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('../images/figure_3_2.png')
            plt.close()
            break
        value = new_value

def figure_3_2_linear_system(policy, epsilon_label):
    """
    Evaluate a given policy by solving the linear system of equations.
    Generate the value function and policy images for the given epsilon label.
    """
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

    # Solve the linear system to compute the value function
    value_function = np.linalg.solve(A, b).reshape(WORLD_SIZE, WORLD_SIZE)

    # Save the value function image
    draw_image(np.round(value_function, decimals=2), f"GW1_Linear_Value_epsilon_{epsilon_label}")
    
    # Save the policy image based on the computed value function
    draw_policy(value_function, f"GW1_Linear_Policy_epsilon_{epsilon_label}")
    
    return value_function


def figure_3_5():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    # value iteration
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('../images/figure_3_5.png')
            plt.close()
            draw_policy(new_value)
            plt.savefig('../images/figure_3_5_policy.png')
            plt.close()
            break
        value = new_value


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
    
def policy_iteration():
    #3(b): Initialize k = 0 - track iteration count
    k = 0 
    
    V = np.zeros((WORLD_SIZE, WORLD_SIZE))  #3(a): V = 0
    policy = initialize_epsilon_greedy_policy(V, epsilon) #3(c): ϵ-greedy policy
    
    while True:
        #Policy evaluation
        while True:
            delta = 0
            for i in range(WORLD_SIZE):
                for j in range(WORLD_SIZE):
                    v = V[i, j]
                    new_value = 0
                    for a, action in enumerate(ACTIONS):
                        (next_i, next_j), reward = step([i, j], action)
                        new_value += policy[i, j, a] * (reward + DISCOUNT * V[next_i, next_j])
                    V[i, j] = new_value
                    delta = max(delta, abs(v - V[i, j]))
            if delta < 1e-4:  #convergence threshold
                break
        
        #Policy improvement
        policy_stable = True
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                old_action = np.argmax(policy[i, j])
                action_values = np.zeros(len(ACTIONS))
                for a, action in enumerate(ACTIONS):
                    (next_i, next_j), reward = step([i, j], action)
                    action_values[a] = reward + DISCOUNT * V[next_i, next_j]

                best_action = np.argmax(action_values) #update to be ϵ-greedy policy
                for a in range(len(ACTIONS)):
                    if a == best_action:
                        policy[i, j, a] = 1 - epsilon + (epsilon / len(ACTIONS))
                    else:
                        policy[i, j, a] = epsilon / len(ACTIONS)

                if old_action != best_action:
                    policy_stable = False
            
        
        k += 1  #3(b): Track iterations
        print(f"Iteration k = {k}")  #iteration count
        
        if policy_stable:
            break
    
    return policy, V

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
        print(f"Testing with epsilon = {epsilon}...")
        V = np.zeros((WORLD_SIZE, WORLD_SIZE))  # Initial value function
        test_policy = get_epsilon_greedy_policy(V, epsilon)  # Compute epsilon-greedy policy
        figure_3_2_linear_system(test_policy, f"{epsilon:.1f}")  # Evaluate the policy

