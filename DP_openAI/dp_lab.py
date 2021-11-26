
"""
# DP Lab 
## Group Members
### Riekert Holder: 2517888
### Justin Abrams: 2483251
### Ashton Naidoo: 2519631
### Humbulani Colbert Nekhumbe: 2340639
"""

import numpy as np
from environments.gridworld import GridworldEnv
import time
import matplotlib.pyplot as plt

def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """

    Vec = np.zeros(env.observation_space.n)
    while True:
        d = 0
        for state in range(env.observation_space.n):
            v = 0
            for a, a_prob in enumerate(policy[state]):
                for prob, state_next, reward, done in env.P[state][a]:
                    v += a_prob * prob * (reward + discount_factor * Vec[state_next])
            d = max(d, np.abs(v - Vec[state]))
            Vec[state] = v
        if d < theta:
            break
    return np.array(Vec)

    #raise NotImplementedError


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        Exp_Val = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for prob, state_next, reward, done in env.P[state][a]:
                Exp_Val[a] += prob * (reward + discount_factor * V[state_next])
        return Exp_Val

    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        V = policy_evaluation_fn(env,policy, discount_factor)
        policy_corr = True
        for state in range(env.observation_space.n):
            all_actions = one_step_lookahead(state, V)      
            if np.argmax(policy[state]) != np.argmax(all_actions):
                policy_corr = False
            policy[state] = np.eye(env.action_space.n)[np.argmax(all_actions)]        
        if policy_corr:
            return policy, V


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """
        Exp_Val = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for p, state_next, r, done in env.P[state][action]:
                Exp_Val[action] += p * (r + discount_factor * V[state_next])
        return Exp_Val

    V = np.zeros(env.observation_space.n)
    while True:
        d = 0
        for state in range(env.observation_space.n):
            Exp_Val = one_step_lookahead(state, V)
            d = max(d, np.abs(np.max(Exp_Val) - V[state]))
            V[state] = np.max(Exp_Val)
        if d < theta:
            break
    
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for state in range(env.observation_space.n):
        Exp_Val = one_step_lookahead(state,V)
        policy[state, np.argmax(Exp_Val)] = 1.0
    return policy, V

def text_print(array):
    # (0=up, 1=right, 2=down, 3=left)
    for x in range(len(array)):
        for y in range(len(array[0])):
            if (x == 4 and y == 4):
                print("o", end="")
            elif (array[x][y] == 0):
                print("U", end="")
            elif (array[x][y] == 1):
                print("R", end="")
            elif (array[x][y] == 2):
                print("D", end="")
            elif (array[x][y] == 3):
                print("L", end="")
        print()

def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)
    state = env.reset()
    print("")
    env.render()
    print("")

    # generate random policy
    rp = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # evaluate random policy
    v = []
    v = policy_evaluation(env,rp)

    # print state value for each state, as grid shape
    print("State values for each State:")
    print(v)
    print("")


    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([-106.81, -104.81, -101.37, -97.62, -95.07,
                           -104.81, -102.25, -97.69, -92.40, -88.52,
                           -101.37, -97.69, -90.74, -81.78, -74.10,
                           -97.62, -92.40, -81.78, -65.89, -47.99,
                           -95.07, -88.52, -74.10, -47.99, 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")
    # use  policy improvement to compute optimal policy and state values
    policy, v = [], []  # call policy_iteration
    policy , v = policy_iteration(env)

    # Print out best action for each state in grid shape
    print("Best action of each State:")
    text_print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

    # print state value for each state, as grid shape
    print("State value fo each state:")
    print(v.reshape(env.shape))
    print("")
    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")
    # use  value iteration to compute optimal policy and state values 
    policy, v = [], []  # call value_iteration
    policy, v = value_iteration(env)
    
    # Print out best action for each state in grid shape 
    print("Best action for each state:")
    text_print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")


    # print state value for each state, as grid shape 
    print("State value for each state:")
    print(v.reshape(env.shape))
    print("")

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                           -7., -6., -5., -4., -3.,
                           -6., -5., -4., -3., -2.,
                           -5., -4., -3., -2., -1.,
                           -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)


if __name__ == "__main__":
    main()

env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)



X = np.logspace(-0.2, 0, num=30)
Y_p = list()
Y_v = list()
for x in X:
    total_p = 0
    total_v = 0
    for i in range(10):
        start = time.time()
        policy , v = policy_iteration(env, discount_factor=x)
        end = time.time()
        total_p += (end - start)
        
        start = time.time()
        policy , v = value_iteration(env, discount_factor=x)
        end = time.time()
        total_v += (end - start)
        
    Y_p.append(total_p/10.0)
    Y_v.append(total_v/10.0)
    
plt.plot(X, Y_p, label="Policy Iteration")
plt.plot(X, Y_v, label="Value Iteration")
plt.legend()
plt.show()

