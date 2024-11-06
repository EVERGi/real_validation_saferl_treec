from copy import deepcopy
from treec.prune_tree import prune_trees


from treec.evaluation import pygmo_eval, individual_to_trees
from treec.logger import TreeLogger

import pygmo as pg
import numpy as np

import ctypes


class RestartBFE(pg.mp_bfe):
    def __init__(self):
        super().__init__()
        self.start_check = 0

    def __call__(self, prob, dvs):
        problem = prob.extract(pygmo_eval)
        stop_dict = problem.stop_dict
        stop_func = stop_dict["stop_func"]
        stop_info = stop_dict["stop_info"]
        algo_id = stop_dict["algo_id"]
        interrupt_func, new_start = stop_func(algo_id, stop_info, self.start_check)
        self.start_check = new_start
        if interrupt_func:
            raise InterruptedError("End of this optimisation")
        return super().__call__(prob, dvs)


def cmaes_stop_criteria(algo_id, stop_info, start_check):
    algo = ctypes.cast(algo_id, ctypes.py_object).value

    pop_size = stop_info["pop_size"]
    dimension = stop_info["dimension"]

    limit_same_score = int(10 + np.ceil(30 * dimension / pop_size))

    stop_optimisation = False
    uda = algo.extract(pg.cmaes)
    log = uda.get_log()

    score_list = [i[2] for i in log]

    while start_check + limit_same_score <= len(score_list):
        test_range = score_list[start_check : start_check + limit_same_score]
        start_check += 1
        NUM_SIGN = 5
        min_sign = np.format_float_positional(
            min(test_range),
            precision=NUM_SIGN,
            unique=False,
            fractional=False,
            trim="k",
        )
        max_sign = np.format_float_positional(
            max(test_range),
            precision=NUM_SIGN,
            unique=False,
            fractional=False,
            trim="k",
        )
        if min_sign == max_sign:
            stop_optimisation = True
            break

    return stop_optimisation, start_check


def cmaes_restart_train(params_evaluation, algo_params):
    dimension = algo_params["dimension"]
    logger = params_evaluation["logger"]
    pop_size = int(4 + np.floor(3 * np.log(dimension)))
    global_stop = False

    global_stop_param = 10**4 * dimension
    while not global_stop:
        print(f"Restarting with population size: {pop_size}")
        algo_params["pop_size"] = pop_size

        try:
            tree_train(params_evaluation, algo_params, cmaes_stop_criteria)
        except InterruptedError:
            pass

        episode_count, best_score = logger.read_best_score_episode_count()
        print(f"Total number of function evaluations: {episode_count}")
        print(f"New best score: {best_score}")
        if episode_count > global_stop_param:
            global_stop = True
        pop_size = pop_size * 2


def tree_train(params_evaluation, algo_params, stop_func=None):
    gen = algo_params["gen"]
    dimension = algo_params["dimension"]
    if "pop_size" in algo_params.keys():
        pop_size = algo_params["pop_size"]
    else:
        pop_size = int(4 + np.floor(3 * np.log(dimension)))

    if "single_threaded" in algo_params.keys():
        single_threaded = algo_params["single_threaded"]
    else:
        single_threaded = False

    if "pygmo_algo" in algo_params.keys():
        pygmo_algo = algo_params["pygmo_algo"]
        if pygmo_algo == "pso_gen":
            uda = pg.pso_gen(gen=gen)
        elif pygmo_algo == "moead_gen":
            uda = pg.moead_gen(gen=gen)
        elif pygmo_algo == "cmaes":
            uda = pg.cmaes(gen=gen)
    else:
        uda = pg.cmaes(gen=gen)
    if stop_func is not None:
        b = pg.bfe(RestartBFE())
    else:
        b = pg.bfe()

    if not single_threaded:
        uda.set_bfe(b)

    algo = pg.algorithm(uda)
    algo.set_verbosity(1)

    if stop_func is not None:
        stop_info = {"dimension": dimension, "pop_size": pop_size}
        stop_dict = {
            "stop_func": stop_func,
            "stop_info": stop_info,
            "algo_id": id(algo),
        }
    else:
        stop_dict = None

    prob = pygmo_eval(dimension, params_evaluation, stop_dict)

    if single_threaded:
        prob.set_single_threaded()

    if single_threaded:
        pop = pg.population(prob, size=pop_size)
    else:
        pop = pg.population(prob, size=pop_size, b=b)

    evolve(algo, pop)


def evolve(algo, pop):
    algo.evolve(pop)


def tree_validate(params_valid, training_folder, params_prune=None):
    if params_prune is None:
        params_prune = deepcopy(params_valid)
    eval_func = params_valid["eval_func"]
    logger = params_valid["logger"]

    model = TreeLogger.get_best_model(training_folder + "models/")
    individual = model["individual"]
    all_nodes_visited = model["all_nodes_visited"]
    trees = individual_to_trees(params_valid, individual)

    trees, _ = prune_trees(trees, all_nodes_visited, params_prune)

    result, all_nodes_visited = eval_func(trees, params_valid)

    model_valid = dict()
    model_valid["individual"] = individual
    model_valid["all_nodes_visited"] = all_nodes_visited
    model_valid["trees"] = trees

    if logger is not None:
        better_score = logger.episode_eval_log(model_valid, result)
        if better_score is not None:
            logger.save_model(model_valid, better_score)

            logger.save_tree_dot(trees, all_nodes_visited, better_score)

    print("Validation score: ", result)
