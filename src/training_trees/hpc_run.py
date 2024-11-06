from train import train_and_valid_house
import sys
import time
import random

def cmaes_restart_hpc(run_seed):
    pop_size = 10*2**int(run_seed)
    while True:
        train_and_valid_house(1)

def cmaes_restart_internal(tree_nodes):
    tree_nodes = int(tree_nodes)
    params_change = {"tree_nodes":tree_nodes}
    train_and_valid_house(1, params_change=params_change, do_restart_train=True)

def pso_train(house_num):
    house_num = int(house_num)
    tree_nodes = 20
    params_change = {"tree_nodes":tree_nodes, "pygmo_algo": "pso_gen", "pop_size": 1000, "gen": 600}
    train_and_valid_house(house_num, params_change=params_change)

if __name__ == "__main__":
    rand_start = 20*random.random()
    time.sleep(rand_start)
    pso_train(sys.argv[1])
