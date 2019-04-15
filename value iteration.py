# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:09:34 2019

@author: Ankit Goyal
"""
import numpy as np
import gym
from gym import wrappers
import time
import seaborn
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab



def run_episode(env, policy, gamma = 1.0, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.
    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.
    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
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


def evaluate_policy(env, policy, gamma = 1.0,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return np.mean(scores)

def extract_policy(v, gamma = 1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.nS)
    for s in range(env.nS):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma = 1.0):
    err=[]
    k=[]
    t=[]
    """ Value-iteration algorithm """
    v = np.zeros(env.nS)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        start=time.time()
        for s in range(env.nS):
            q_sa = [sum([p*(r + prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
            v[s] = max(q_sa)
        end=time.time()
        ti=end-start
        t.append(ti)
        print(t)
        err.append(np.sum(np.fabs(prev_v-v)))
        k.append(i)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print ('Value-iteration converged at iteration# %d.' %(i+1))
            err=np.array(err)
            k=np.array(k)
            t=np.array(t)
            s=np.sum(t)
            #line1, = plt.plot(k,err,color='r',label='error')
            line2,=plt.plot(k,t,color='g',label='time')
            #plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
            plt.ylabel('time')
            plt.xlabel('iteration')
            plt.show()
            break
    return s


if __name__ == '__main__':
    env_name  = 'FrozenLake-v0'
    gamma = 0.9
    env = gym.make(env_name)
    #optimal_v = value_iteration(env, gamma);
    #policy = extract_policy(optimal_v, gamma)
    #policy_str=print_policy(policy,lake_env.action_names)
    #ps=[]
    #for elem in policy:
     #   ps.append(elem)
    #reshaped_policy=np.reshape(ps,(8,8))
    #print(tabulate(reshaped_policy,tablefmt='latex'))
    #f, ax = plt.subplots(figsize=(11, 9))
    #cmap = seaborn.diverging_palette(220, 10, as_cmap=True)
    #reshaped=np.reshape(optimal_v,(8,8))
    #seaborn.heatmap(reshaped, cmap=cmap, vmax=1.1,
     #       square=True, xticklabels=8+1, yticklabels=8+1,
      #      linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    #plt.savefig('1c.png',bbox_inches='tight')
    #np.savetxt('1gpolicy.csv',reshaped,delimiter=',')
    #print(policy)
    #env.render()
    #policy_score = evaluate_policy(env, policy, gamma, n=1000)
    #print('Policy average score = ', policy_score)
    s=[]
    for i in range(100):
        steps=value_iteration(env, gamma = gamma)
        s.append(steps)
    s=np.array(s)
    (mu, sigma) = norm.fit(s)
    n, bins, patches = plt.hist(s, 60, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('Steps')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ steps:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
    plt.grid(True)
    plt.show()