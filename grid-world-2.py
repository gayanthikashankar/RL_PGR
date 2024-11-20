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

def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5

    # Transition probabilities for each action
    transition_probs = {
        'left': [(0, -1), (0, 1), (-1, 0), (1, 0)],  
        'up': [(-1, 0), (1, 0), (0, -1), (0, 1)],    
        'right': [(0, 1), (0, -1), (1, 0), (-1, 0)],  
        'down': [(1, 0), (-1, 0), (0, -1), (0, 1)]    
    }

    # Probabilities for each action direction (85% for action, 5% for others)
    action_probabilities = {
        'left': [0.85, 0.05, 0.05, 0.05],  
        'up': [0.05, 0.85, 0.05, 0.05],    
        'right': [0.05, 0.05, 0.85, 0.05],  
        'down': [0.05, 0.05, 0.05, 0.85]    
    }

    # State to index
    x, y = state

    # Determine action name
    if np.array_equal(action, ACTIONS[0]):
        action_name = 'left'
    elif np.array_equal(action, ACTIONS[1]):
        action_name = 'up'
    elif np.array_equal(action, ACTIONS[2]):
        action_name = 'right'
    else:  
        action_name = 'down'

    # Get transition probabilities and possible directions
    action_transitions = transition_probs[action_name]
    action_probs = action_probabilities[action_name]

    # Sample next state based on probabilities
    chosen_action = np.random.choice(4, p=action_probs)
    dx, dy = action_transitions[chosen_action]

    # Calculate the next state
    next_state = [x + dx, y + dy]

    # Within bounds constraint
    if next_state[0] < 0 or next_state[0] >= WORLD_SIZE or next_state[1] < 0 or next_state[1] >= WORLD_SIZE:
        next_state = state
        reward = -1.0
    else:
        reward = 0  # No reward if the action doesn't result in a boundary hit

    return next_state, reward

def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):
        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"
        
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')
        

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals=[]
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0],next_state[1]])

        best_actions=np.where(next_vals == np.max(next_vals))[0]
        val=''
        for ba in best_actions:
            val+=ACTIONS_FIGS[ba]
        
        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"
        
        tb.add_cell(i, j, width, height, text=val,
                loc='center', facecolor='white')

    # Row and column labels...
    for i in range(len(optimal_values)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                   edgecolor='none', facecolor='none')

    ax.add_table(tb)

def figure_3_2():
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
        # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    new_value[i, j] += ACTION_PROB * (reward + DISCOUNT * value[next_i, next_j])
        if np.sum(np.abs(value - new_value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('./images/GW2figure_3_2.png')
            plt.close()
            break
        value = new_value

def figure_3_2_linear_system():
    A = -1 * np.eye(WORLD_SIZE * WORLD_SIZE)
    b = np.zeros(WORLD_SIZE * WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            s = [i, j]  #current state
            index_s = np.ravel_multi_index(s, (WORLD_SIZE, WORLD_SIZE))
            for a in ACTIONS:
                s_, r = step(s, a)
                index_s_ = np.ravel_multi_index(s_, (WORLD_SIZE, WORLD_SIZE))

                A[index_s, index_s_] += ACTION_PROB * DISCOUNT
                b[index_s] -= ACTION_PROB * r

    solution = np.linalg.solve(A, b)
    solution = solution.reshape((WORLD_SIZE, WORLD_SIZE))

    draw_image(np.round(solution, decimals=2))
    plt.savefig('./images/GW2figure_3_2_linear_system.png')
    plt.close()

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
                    values.append(reward + DISCOUNT * value[next_i, next_j])
                new_value[i, j] = np.max(values)
        if np.sum(np.abs(new_value - value)) < 1e-4:
            draw_image(np.round(new_value, decimals=2))
            plt.savefig('./images/GW2figure_3_5.png')
            plt.close()
            draw_policy(new_value)
            plt.savefig('./images/GW2figure_3_5_policy.png')
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

def evaluate_policy(policy, epsilon):
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

    # solve system of linear equations
    value_function = np.linalg.solve(A, b).reshape(WORLD_SIZE, WORLD_SIZE)

    # draw and save results
    draw_image(np.round(value_function, decimals=2))
    plt.savefig('./images/GW2evaluated_policy_value.png')
    plt.close()

    draw_policy(value_function)
    plt.savefig('./images/GW2evaluated_policy.png')
    plt.close()

    return value_function

def policy_iteration(epsilon):
    # Initialize k = 0 - track iteration count
    k = 0 
    
    V = np.zeros((WORLD_SIZE, WORLD_SIZE))  # Initial value function
    policy = get_epsilon_greedy_policy(V, epsilon)
    
    while True:
        # Policy evaluation
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
            if delta < 1e-4:  # convergence threshold
                break
        
        # Policy improvement
        policy_stable = True
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                old_action = np.argmax(policy[i, j])
                action_values = np.zeros(len(ACTIONS))
                for a, action in enumerate(ACTIONS):
                    (next_i, next_j), reward = step([i, j], action)
                    action_values[a] = reward + DISCOUNT * V[next_i, next_j]

                best_action = np.argmax(action_values)
                for a in range(len(ACTIONS)):
                    if a == best_action:
                        policy[i, j, a] = 1 - epsilon + (epsilon / len(ACTIONS))
                    else:
                        policy[i, j, a] = epsilon / len(ACTIONS)

                if old_action != best_action:
                    policy_stable = False
            
        k += 1  # Track iterations
        print(f"Iteration k = {k}")
        
        if policy_stable:
            break
    
    return policy, V

if __name__ == '__main__':
    #Set random seed for reproducibility
    np.random.seed(42)
    
    figure_3_2_linear_system()  
    figure_3_2()  
    figure_3_5()  

    for epsilon in [0.2, 0.0]:
        print(f"Running policy iteration with epsilon = {epsilon}")
        policy, V = policy_iteration(epsilon)
        evaluate_policy(policy, epsilon)