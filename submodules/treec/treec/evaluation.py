from treec.tree import (
    BinaryTreeFixed,
)

import ctypes


def evaluate_with_leafs(individual, params_evaluation):
    individual = list(individual)
    eval_func = params_evaluation["eval_func"]
    logger = params_evaluation["logger"]

    for i, indiv in enumerate(individual):
        if indiv < 0:
            individual[i] = 0
        elif indiv >= 1:
            individual[i] = 0.9999

    trees = individual_to_trees(params_evaluation, individual)

    result, all_nodes_visited = eval_func(trees, params_evaluation)

    if logger is not None:
        model = {
            "individual": individual,
            "all_nodes_visited": all_nodes_visited,
            "trees": trees,
        }
        better_score = logger.episode_eval_log(model, result)
        if better_score is not None:
            logger.save_model(model, better_score)
            logger.save_tree_dot(trees, all_nodes_visited, better_score)

    return result, all_nodes_visited


def evaluate(individual, params_evaluation):
    result, _ = evaluate_with_leafs(individual, params_evaluation)
    return (result,)


def individual_to_trees(params_evaluation, individual):
    TreeStruct = BinaryTreeFixed
    input_func = params_evaluation["input_func"]
    action_names = params_evaluation["action_names"]
    num_trees = len(action_names)

    _, feature_info = input_func(params_evaluation)

    trees = list()
    sub_indiv_size = len(individual) // num_trees
    for i in range(num_trees):
        # tree_feature_info = [feat[0] for feat in feature_info[i]]
        tree_raw = individual[i : i + sub_indiv_size]
        trees.append(TreeStruct(tree_raw, feature_info[i], action_names[i]))
    return trees


class pygmo_eval:
    def __init__(self, dimension, params_evaluation, stop_dict=None):
        self.dim = dimension
        self.params_evaluation = params_evaluation
        self.stop_dict = stop_dict
        self.single_threaded = False
        self.start_check = 0

    def get_bounds(self):
        return (
            [0.0 for _ in range(self.dim)],
            [1.0 for _ in range(self.dim)],
        )

    def set_single_threaded(self):
        self.single_threaded = True

    def fitness(self, x):
        if self.stop_dict is not None and self.single_threaded:
            algo_id = self.stop_dict["algo_id"]
            stop_info = self.stop_dict["stop_info"]
            stop_func = self.stop_dict["stop_func"]
            interrupt_func, new_start = stop_func(algo_id, stop_info, self.start_check)
            self.start_check = new_start
            if interrupt_func:
                raise InterruptedError("End of this optimisation")

        result = evaluate(x, self.params_evaluation)
        return [-result[0]]
