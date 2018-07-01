############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Local Search-2-opt Stochastic

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Local_Search-2-opt_Stochastic, File: Python-MH-Local Search-2-opt-Stochastic.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Local_Search-2-opt_Stochastic>

############################################################################

# Required Libraries
import pandas as pd
import random
import copy

# Function: Distance
def distance_calc(Xdata, route):
    distance = 0
    for k in range(0, len(route[0])-1):
        m = k + 1
        distance = distance + Xdata.iloc[route[0][k]-1, route[0][m]-1]            
    return distance

# Function: 2_opt Stochastic
def local_search_2_opt_stochastic(Xdata, city_tour, recursive_seeding = 1):
    count = 0
    city_list = copy.deepcopy(city_tour)
    while (count < recursive_seeding):
        best_route = copy.deepcopy(city_list)
        seed = copy.deepcopy(city_list)        
        for i in range(0, len(city_list[0]) - 2):
            for j in range(i+1, len(city_list[0]) - 1):
                m, n  = random.sample(range(0, len(city_tour[0])-1), 2)
                if (m > n):
                    m, n = n, m
                best_route[0][m:n+1] = list(reversed(best_route[0][m:n+1]))           
                best_route[0][-1]  = best_route[0][0]              
                best_route[1] = distance_calc(Xdata, best_route)                     
                if (best_route[1] < city_list[1]):
                    city_list[1] = copy.deepcopy(best_route[1])
                    for k in range(0, len(city_list[0])): 
                        city_list[0][k] = best_route[0][k]          
                best_route = copy.deepcopy(seed)
        count = count + 1  
        print("Iteration = ", count, "->", city_list)
    return city_list
######################## Part 1 - Usage ####################################

X = pd.read_csv('Python-MH-Local Search-2-opt-Stochastic-Dataset-01.txt', sep = '\t') #17 cities => Optimum = 2085

cities = [[   1,  2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,   1   ], 4722]
ls2opts = local_search_2_opt_stochastic(X, city_tour = cities, recursive_seeding = 100)

