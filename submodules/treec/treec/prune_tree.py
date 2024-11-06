from collections import Counter
from copy import deepcopy
import copy


def prune_unvisited_nodes(binarytree, leafs_visited_index):
    leafs_visited_node = [binarytree.node_stack[i] for i in leafs_visited_index]
    leafs_count = Counter(leafs_visited_node)

    if len(leafs_count) == 1:
        only_node = list(leafs_count.keys())[0]
        binarytree.root_node = only_node
        only_node.root = True
        binarytree.recalculate_node_stack()
        return [0 for _ in leafs_visited_index]
    left_node = binarytree.root_node.left_node
    right_node = binarytree.root_node.right_node

    remove_or_not_node(left_node, leafs_count, binarytree)
    remove_or_not_node(right_node, leafs_count, binarytree)

    new_index = {node: i for i, node in enumerate(binarytree.node_stack)}
    new_leafs_visited_index = [new_index[i] for i in leafs_visited_node]
    return new_leafs_visited_index


def remove_or_not_node(cur_node, leafs_count, binarytree):
    parent_node = binarytree.get_parent(cur_node)

    if cur_node.feature == -1:
        if leafs_count[cur_node] == 0:
            if parent_node.left_node is cur_node:
                node_to_keep = parent_node.right_node
            else:
                node_to_keep = parent_node.left_node

            if parent_node.root:
                binarytree.root_node = node_to_keep
                node_to_keep.root = True

            else:
                grand_parent_node = binarytree.get_parent(parent_node)
                if grand_parent_node.left_node is parent_node:
                    grand_parent_node.left_node = node_to_keep
                else:
                    grand_parent_node.right_node = node_to_keep
            binarytree.recalculate_node_stack()
    else:
        left_node = cur_node.left_node
        right_node = cur_node.right_node
        remove_or_not_node(left_node, leafs_count, binarytree)
        remove_or_not_node(right_node, leafs_count, binarytree)


def prune_same_action_nodes(binarytree, leafs_visited_index):
    leafs_visited_node = [binarytree.node_stack[i] for i in leafs_visited_index]

    binarytree.recalculate_depth()
    max_depth = max([node.depth for node in binarytree.node_stack])

    for depth in reversed(range(max_depth)):
        for node in binarytree.node_stack:
            if node.left_node is None or node.right_node is None or node.depth != depth:
                continue

            children_are_leafs = (
                node.left_node.feature == -1 and node.right_node.feature == -1
            )
            same_value = node.left_node.value == node.right_node.value

            if children_are_leafs and same_value:
                if node.root:
                    binarytree.root_node = node.left_node
                    node.left_node.root = True
                else:
                    parent = binarytree.get_parent(node)
                    if parent.right_node is node:
                        parent.right_node = node.left_node
                    else:
                        parent.left_node = node.left_node

                leafs_visited_node = [
                    node.left_node if i is node.right_node else i
                    for i in leafs_visited_node
                ]
                binarytree.recalculate_depth()

    new_index = {node: i for i, node in enumerate(binarytree.node_stack)}
    new_leafs_visited_index = [new_index[i] for i in leafs_visited_node]

    return new_leafs_visited_index


def prune_score(trees, params_pruning, pruning_tol=None):
    eval_score, all_nodes_visited = evaluate_trees(trees, params_pruning)
    if pruning_tol is None:
        # 1 percent tolerance of the score
        score_tol = eval_score - abs(eval_score / 100)
    else:
        score_tol = eval_score - pruning_tol

    common_sorted = list()
    for tree_num, binarytree in enumerate(trees):
        leafs_visited_index = [j[tree_num] for j in all_nodes_visited]
        leafs_visited_node = [binarytree.node_stack[j] for j in leafs_visited_index]

        leafs_count = Counter(leafs_visited_node)
        sorted_count = sorted(leafs_count.items(), key=lambda x: x[1])
        sorted_count = [[node, count, tree_num] for node, count in sorted_count]
        common_sorted += sorted_count

    common_sorted = sorted(common_sorted, key=lambda x: x[1])

    for j, sorted_tuple in enumerate(common_sorted):
        node = sorted_tuple[0]
        node.prune_num = j

    for prune_num, count_var in enumerate(common_sorted):
        tree_num = count_var[2]
        new_tree = deepcopy(trees[tree_num])
        for node in new_tree.node_stack:
            if hasattr(node, "prune_num") and node.prune_num == prune_num:
                cur_node = node
                break

        parent_node = new_tree.get_parent(cur_node)

        if cur_node.root:
            continue

        if parent_node.left_node is cur_node:
            node_to_keep = parent_node.right_node
        else:
            node_to_keep = parent_node.left_node

        if parent_node.root:
            new_tree.root_node = node_to_keep
            node_to_keep.root = True

        else:
            grand_parent_node = new_tree.get_parent(parent_node)
            if grand_parent_node.left_node is parent_node:
                grand_parent_node.left_node = node_to_keep
            else:
                grand_parent_node.right_node = node_to_keep
        new_tree.recalculate_depth()
        trees_eval = copy.copy(trees)
        trees_eval[tree_num] = new_tree
        # print(f"Node stack length tree batt: {len(trees[0].node_stack)}")
        # print(f"Node stack length tree charg: {len(trees[1].node_stack)}")
        new_eval_score, new_all_nodes_visited = evaluate_trees(
            trees_eval, params_pruning
        )

        if new_eval_score > score_tol:
            trees = copy.copy(trees_eval)
            all_nodes_visited = new_all_nodes_visited

    return trees, all_nodes_visited


def evaluate_trees(trees, params_pruning):
    eval_func = params_pruning["eval_func"]
    eval_score, all_nodes_visited = eval_func(trees, params_pruning)
    return eval_score, all_nodes_visited


def prune_tree(binarytree, leafs_visited_index):
    leafs_visited_index = prune_unvisited_nodes(binarytree, leafs_visited_index)

    new_leafs_visited_index = prune_same_action_nodes(binarytree, leafs_visited_index)

    return new_leafs_visited_index


def prune_trees(trees, all_nodes_visited, params_pruning):
    for i, tree in enumerate(trees):
        leafs_visited_index = [j[i] for j in all_nodes_visited]
        prune_tree(tree, leafs_visited_index)

    trees, all_nodes_visited = prune_score(trees, params_pruning)

    return trees, all_nodes_visited
