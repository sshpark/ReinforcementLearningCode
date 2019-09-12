import gym
import time
import numpy as np

env = gym.make('FrozenLake-v0')

def value_iteration(env, gamma = 1.0):
    value_table = np.zeros(env.observation_space.n)
    no_of_iterations = 100000
    threshold = 1e-20
    optimal_policy = np.zeros(env.observation_space.n)

    for i in range(no_of_iterations):
        update_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob * (reward_prob + gamma * 
                        update_value_table[next_state])))
                Q_value.append(np.sum(next_states_rewards))

            value_table[state] = max(Q_value)
            optimal_policy[state] = np.argmax(Q_value)

        if np.sum(np.fabs(update_value_table - value_table)) <= threshhold:
            print('Value-iteration converged at iteration# %d.' % (i+1))
            break

    return value_table, optimal_policy

optimal_value_function, optimal_policy = value_iteration(env = env, gamma = 1.0)

print(optimal_policy)