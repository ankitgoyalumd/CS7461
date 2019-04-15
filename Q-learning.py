# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 01:24:06 2019

@author: Ankit Goyal
"""

import gym
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
# 1. Load Environment and Q-table structure
env = gym.make('FrozenLake8x8-v0')
eta = .628
gma = .9
epis = 10000

def Q_learn(env,eta,gma,epis):
    Q = np.zeros([env.observation_space.n,env.action_space.n])
# env.obeservation.n, env.action_space.n gives number of states and action in env loaded
# 2. Parameters of Q-leanring
   
    rev_list = [] # rewards per episode calculate
    k=[]
# 3. Q-learning Algorithm
    start=time.time()
    for i in range(epis):
    # Reset environment
        s = env.reset()
        rAll = 0
        d = False
        j = 0
    #The Q-Table learning algorithm
        while j < 99:
            #env.render()
            j+=1
            # Choose action from Q table
            a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            #print(a)
            #Get new state & reward from environment
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            if d == True:
                rAll+=r
                break
            end=time.time()
            rev_list.append(rAll)
            #env.render()
            k.append(i)
    print(end-start)
    rev_list=np.array(rev_list)
    k=np.array(k)
    line2,=plt.plot(k,rev_list,color='g',label='performance')
    plt.ylabel('performance')
    plt.xlabel('iteration')
    plt.show()
    print ("Reward Sum on all episodes " + str(sum(rev_list)/epis))
    print ("Final Values Q-Table")
    print (Q)
    return end-start

Q_learn(env,eta,gma,epis)

#Q_learn(env,0.001,0.9,100000)# Reset environment
#for i in range(6000):
 #   s = env.reset()
  #  d = False
# The Q-Table learning algorithm
   # Q=Q_learn(env,0.01,0.9,10000)
    #while d != True:
     #   env.render()
    # Choose action from Q table
      #  a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
    #Get new state & reward from environment
       # s1,r,d,_ = env.step(a)
    #Update Q-Table with new knowledge
        #Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
        #s = s1
        #print(i)
# Code will stop at d == True, and render one state before it
s=[]
#for i in range(30):
 #   t=Q_learn(env,eta,gma, epis)
  #  s.append(t)
#s=np.array(s)
#(mu, sigma) = norm.fit(s)
#n, bins, patches = plt.hist(s, 60, normed=1, facecolor='green', alpha=0.75)
#plt.xlabel('Steps')
#plt.ylabel('Probability')
#plt.title(r'$\mathrm{Histogram\ of\ steps:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
#plt.grid(True)
#plt.show()

