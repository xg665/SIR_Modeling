#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import copy
from scipy.integrate import odeint
from scipy.optimize import root_scalar
import random
from bisect import bisect_right
import time
from mpmath import *
from mpl_toolkits import mplot3d
import pickle

def network_iter(G,q,inf_init,N=4000,M=800,b=0.075,b2=0.06,gamma=0.2,pos_sen=0.98,neg_sen=0.85,delay=1,ratio=1): # N: total population, M: number of edge nodes
                                                             # P:testing period, Q: number of nodes being tested
                                                             # sensitivity: test accuracy
    counter = 0
    N_nodes = set(np.arange(0,N,1))

    test_list = np.random.permutation(N)

    edges = random.sample(N_nodes,M)                     # randomly sample M edges nodes (boundary conditions) from whole population

    infected= inf_init                # randomly select 1 node from M edge nodes to be first infected

    suscep=[s for s in N_nodes if s not in infected]     # rest are susceptible

    removed=[]

    quarantine = []

    for inf in infected:                           #initialize node attribute
        G.nodes[inf]['status']='infected'
        G.nodes[inf]['inf_dur'] = 0
        G.nodes[inf]['test_day'] = None
        G.nodes[inf]['qua_dur'] = None
        G.nodes[inf]['result'] = None

    for sus in suscep:
        G.nodes[sus]['status']='susceptible'
        G.nodes[sus]['test_day'] = None
        G.nodes[sus]['qua_dur'] = None
        G.nodes[sus]['result'] = None
        G.nodes[sus]['inf_dur']= None

    gamma_inverse = 1/gamma
    I = 1
    S = N-I
    R = 0
    Q = 0
    dt = 1
    num_qua_pos = 0
    daily_case = []

    I_record=[1]
    S_record=[N-1]
    R_record=[0]
    Q_record=[0]
    TotalCases = I+R
    finished = False

    while len(infected)>0 and finished == False:
        new_infected = []

        for sus in suscep:
            nei = G[sus].keys()       # get current susceptible's neighbors
            infected_nei = [n for n in nei if G.nodes[n]['status']=='infected']   # get infectious neighbors
            p_infection = 1-np.power((1-b*dt),len(infected_nei))
            inf_status = np.random.binomial(1,p_infection)
            if inf_status==1:
                new_infected.append(sus)
                I=I+1
                S=S-1
                G.nodes[sus]['inf_dur'] = 0

        new_removed = []

        for inf in infected:
            G.nodes[inf]['inf_dur']=G.nodes[inf]['inf_dur']+dt

            if G.nodes[inf]['inf_dur']>=gamma_inverse:
                new_removed.append(inf)
                I=I-1
                R=R+1

        new_infected= list(dict.fromkeys(new_infected))
        new_removed = list(dict.fromkeys(new_removed))

        for re in new_removed:
            infected.remove(re)
            G.nodes[re]['status']='removed'

        for inf in new_infected:
            suscep.remove(inf)
            G.nodes[inf]['status']='infected'
            G.nodes[inf]['inf_dur']=0

        infected.extend(new_infected)
        removed.extend(new_removed)
        test_subjects = divide_test(counter,q,N,test_list,ratio)
        counter += 1

        G,S,I,R,Q,infected,quarantine,removed,suscep,finished = delay_test(G,N,dt,pos_sen,neg_sen,S,I,R,Q,
                                                                            test_subjects,delay,infected,
                                                                            quarantine,removed,suscep,
                                                                            finished,gamma_inverse)

        for qua in quarantine:
            if G.nodes[qua]['status'] == 'quarantined_pos':
                num_qua_pos += 1


        I_record.append(len(infected))
        R_record.append(len(removed))
        S_record.append(len(suscep))
        Q_record.append(len(quarantine))

        if finished:
            detected_day = counter

    TotalCases = len(infected)+len(removed)+num_qua_pos

    return S_record,I_record,R_record,Q_record,TotalCases,counter


def divide_test(counter,q,N,test_list,ratio=1):
    N = int(N * ratio)
    test_list = test_list[:N]
    if (q*counter)%N < (q*(counter+1))%N:
        return test_list[(q*counter)%N:(q*(counter+1))%N]
    elif (q*counter)%N >= (q*(counter+1))%N:
        return list(test_list[(q*counter)%N:])+list(test_list[:(q*(counter+1))%N])

def delay_test(G,N,dt,pos_sen,neg_sen,S,I,R,Q,test_subjects,delay,infected,quarantine,removed,suscep,finished,gamma_inverse):
    for node in range(N):
        if str.isdigit(str(G.nodes[node]['test_day'])):
            G.nodes[node]['test_day'] += dt
        if str.isdigit(str(G.nodes[node]['qua_dur'])):
            G.nodes[node]['qua_dur'] += dt
        if str.isdigit(str(G.nodes[node]['inf_dur'])) and (G.nodes[node]['status'] == 'quarantined_pos'):
            G.nodes[node]['inf_dur'] += dt

    for t in test_subjects:
        if G.nodes[t]['status'] in ['infected', 'removed']:
            true_inf = np.random.binomial(1,pos_sen)
            if true_inf == 1:
                G.nodes[t]['result'] = 'quarantined_pos'
                G.nodes[t]['test_day'] = 0
        if G.nodes[t]['status']=='susceptible':
            true_sus = np.random.binomial(1,neg_sen)
            if true_sus != 1:
                G.nodes[t]['result'] = 'quarantined_neg'
                G.nodes[t]['test_day'] = 0

    for node in range(N):
        if G.nodes[node]['test_day'] == delay:
            if G.nodes[node]['status']=='infected':
                if G.nodes[node]['result'] == 'quarantined_pos':
                    Q += 1
                    I -= 1
                    infected.remove(node)
                    G.nodes[node]['status'] = 'quarantined_pos'
                    quarantine.append(node)
                    G.nodes[node]['qua_dur'] = 0
                    G.nodes[node]['test_day'] = None
                    true_inf = np.random.binomial(1,pos_sen)
                    if true_inf == 1:
                        G.nodes[node]['result'] = 'quarantined_pos'
                    elif true_inf != 1:
                        G.nodes[node]['result'] = None
            if G.nodes[node]['status']=='removed':
                if G.nodes[node]['result'] == 'quarantined_pos':
                    Q += 1
                    R -= 1
                    removed.remove(node)
                    G.nodes[node]['status'] = 'quarantined_pos'
                    G.nodes[node]['qua_dur'] = 0
                    G.nodes[node]['test_day'] = None
                    quarantine.append(node)
                    true_inf = np.random.binomial(1,pos_sen)
                    if true_inf == 1:
                        G.nodes[node]['result'] = 'quarantined_pos'
                    elif true_inf != 1:
                        G.nodes[node]['result'] = None
            if G.nodes[node]['status']=='susceptible':
                if G.nodes[node]['result'] == 'quarantined_neg':
                    Q += 1
                    S -= 1
                    suscep.remove(node)
                    G.nodes[node]['status'] = 'quarantined_neg'
                    G.nodes[node]['qua_dur'] = 0
                    G.nodes[node]['test_day'] = None
                    quarantine.append(node)
                    true_sus = np.random.binomial(1,neg_sen)
                    if true_sus != 1:
                        G.nodes[node]['result'] = 'quarantined_neg'
                    elif true_sus == 1:
                        G.nodes[node]['result'] = None
        if G.nodes[node]['qua_dur'] == delay:
            G.nodes[node]['qua_dur'] = None
            if G.nodes[node]['status'] == G.nodes[node]['result']:
                Q -= 1
                R += 1
                G.nodes[node]['status'] = 'removed'
                G.nodes[node]['result'] = None
                quarantine.remove(node)
                removed.append(node)
                finished = True
            elif G.nodes[node]['status'] == 'quarantined_neg' and (not G.nodes[node]['result']):
                Q -= 1
                S += 1
                G.nodes[node]['status'] = 'susceptible'
                quarantine.remove(node)
                suscep.append(node)
            elif G.nodes[node]['status'] == 'quarantined_pos' and (not G.nodes[node]['result']):
                if G.nodes[node]['inf_dur']>=gamma_inverse:
                    Q -= 1
                    R += 1
                    G.nodes[node]['status'] = 'removed'
                    quarantine.remove(node)
                    removed.append(node)
                elif G.nodes[node]['inf_dur']<gamma_inverse:
                    Q -= 1
                    I += 1
                    G.nodes[node]['status'] = 'infected'
                    quarantine.remove(node)
                    infected.append(node)

    return G,S,I,R,Q,infected,quarantine,removed,suscep,finished


if __name__ == '__main__':

    all_q = np.arange(10,600,20)
    ratio_list = [0.2,0.3,1]

    for ratio in ratio_list:
        all_cases= []
        for i in range(len(all_q)):
            c= []
            for round in range(100):
                with open('G.pickle', 'rb') as f:
                    G = pickle.load(f)

                with open('inf_sample.pickle', 'rb') as f:
                    infectious = pickle.load(f)
                init_inf = random.sample(infectious,1)
                S,I,R,Q,cases,counter= network_iter(G,all_q[i],init_inf,ratio=ratio)
                c.append(cases)
            all_cases.append(np.mean(c))

        with open('testing_among_all_with_ratio'+''.join(str(ratio).split('.'))+'.pickle', 'wb') as f:
            pickle.dump(all_cases,f)

    with open('testing_among_all_with_ratio02.pickle', 'rb') as f:
        decrease1 = pickle.load(f)
    with open('testing_among_all_with_ratio03.pickle', 'rb') as f:
        decrease2 = pickle.load(f)
    with open('testing_among_all_with_ratio1.pickle', 'rb') as f:
        decrease3 = pickle.load(f)

    fig,ax = plt.subplots()
    all_q = np.arange(10,600,20)
    ax.plot(all_q,decrease1,label='20% testing with 1d delay')
    ax.plot(all_q,decrease2,label='30% testing with 1d delay')
    ax.plot(all_q,decrease3,label='100% testing with 1d delay')
    ax.set_title("Part of population have been Testing")
    ax.set_xlabel("Number of people been tested per day")
    ax.set_ylabel("Outbreak sizes")
    ax.legend()
    ax.grid()
