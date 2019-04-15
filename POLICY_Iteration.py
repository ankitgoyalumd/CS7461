# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 23:43:59 2019

@author: Ankit Goyal
"""
import numpy as np
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
import matplotlib.mlab as mlab

g=0.95
def run_episode(env, policy, gamma = g, render = False):
    """ Runs an episode and return the total reward """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma = g, n = 100):
    scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = g):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.nA)
        for a in range(env.nA):
            q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
        policy[s] = np.argmax(q_sa)
    return policy

def compute_policy_v(env, policy, gamma=g):
    """ Iteratively evaluate the value-function under policy.
    Alternatively, we could formulate a set of linear equations in iterms of v[s] 
    and solve them to find the value function.
    """
    v = np.zeros(env.nS)
    eps = 1e-10
    while True:
        prev_v = np.copy(v)
        for s in range(env.nS):
            policy_a = policy[s]
            v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][policy_a]])
        if (np.sum((np.fabs(prev_v - v))) <= eps):
            # value converged
            break
    return v

def policy_iteration(env, gamma = g):
    """ Policy-Iteration algorithm """
    policy = np.random.choice(env.nA, size=(env.nS))  # initialize a random policy
    max_iterations = 200000
    gamma = 1.0
    score=[]
    k=[]
    t=[]
    for i in range(max_iterations):
        start=time.time()
        old_policy_v = compute_policy_v(env, policy, gamma)
        new_policy = extract_policy(old_policy_v, gamma)
        s = evaluate_policy(env, new_policy, gamma = g)
        end=time.time()
        ti=end-start
        t.append(ti)
        score.append(s)
        k.append(i)
        #print(score)
        print(new_policy)
        if (np.all(policy == new_policy)):
            score=np.array(score)
            k=np.array(k)
            t=np.array(t)
            #line1, = plt.plot(k,score,color='r',label='error')
            #line1, = plt.plot(k,t,color='g',label='time',marker='*')
            s=np.sum(t)
            print(s)
            plt.xlabel('Steps')
            plt.ylabel('time(seconds)')
            print(score)
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        policy = new_policy
    return s


if __name__ == '__main__':
    env_name  = 'FrozenLake-v0'
    env = gym.make(env_name)
    #optimal_policy = policy_iteration(env, gamma = g)
    #print(optimal_policy)
    #env.render()
    #scores = evaluate_policy(env, optimal_policy, gamma = g)
    #print('Average scores = ', np.mean(scores))
    s=[]
    for i in range(100):
        steps=policy_iteration(env, gamma = g)
        s.append(steps)
    s=np.array(s)
    (mu, sigma) = norm.fit(s)
    n, bins, patches = plt.hist(s, 60, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Steps')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ steps:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
    plt.grid(True)
    plt.show()

        