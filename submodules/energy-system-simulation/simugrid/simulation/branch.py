import numpy as np
import scipy.optimize as opt


class Branch:
    def __init__(self, nodes_index):
        # Intrinsic characteristics
        self.length = None
        self.resistivity = None
        self.diameter = None
        self.resistance = 10**-20

        self.max_power_electrical = 1000000
        self.max_power_heating = 1000000

        self.nodes_index = nodes_index
        self.name = str(nodes_index[0]) + "to" + str(nodes_index[1])

        self.init_var = dict()

        # Computed values
        self.power_electrical = 0
        self.power_heating = 0

        self.losses_electrical = 0
        self.losses_heating = 0

    def set_attributes(self, var_dict):
        """
        Set asset attributes from dictionary and checks if the environment
            values are present

        :param var_dict: dictionary with attribute name as key and attribute
            value as value
        :type var_dict: dict

        :return: checks if the simulation can run for the given attributes.
            Return True if it can run. False if not
        :rtype: bool
        """

        # check if all attributes are an atrribute of the asset

        attr_of_asset = [i for i in var_dict.keys() if not hasattr(self, i)]

        for key in attr_of_asset:
            var_dict.pop(key)

        for var_name, value in var_dict.items():
            setattr(self, var_name, value)

        for key, value in var_dict.items():
            self.init_var[key] = value

        if attr_of_asset:
            message = "no such attributes {} in branch {}:".format(
                ", ".join(attr_of_asset), self.name
            )
            print("\033[93m" + message + "\033[0m")

        if None not in [self.length, self.resistivity, self.diameter]:
            self.resistance = self.length * self.resistivity / self.diameter


def get_limit_to_max_branch_power(microgrid, power_changes=dict()):
    branches = microgrid.branches
    power_branches = get_branches_power(microgrid, power_changes)

    limit_to_max_power = list()
    for i, branch in enumerate(branches):
        branch_max_pow_diff = branch.max_power_electrical - abs(power_branches[i])
        limit_to_max_power.append(branch_max_pow_diff)

    return limit_to_max_power


def get_branches_power(microgrid, power_changes=dict()):
    nodes_power = list()
    for node in microgrid.nodes:
        power_node = 0
        for asset in node.assets:
            power_node += asset.power_output.electrical
            if asset in power_changes.keys():
                power_node += power_changes[asset]
        nodes_power.append(power_node)

    branches = microgrid.branches

    power_branches = calculate_branches_power(branches, nodes_power)

    return power_branches


def calculate_branches_power(branch_list, nodes_power):
    if len(branch_list) == 1:
        sol_branch = [nodes_power[0]]
        return sol_branch

    branch_indexes = []
    # Line resistance values
    branch_resistance = []

    for branch in branch_list:
        branch_indexes.append(branch.nodes_index)
        branch_resistance.append(branch.resistance)

    # Transform into matrices
    r = np.zeros((len(branch_resistance), len(branch_resistance)))
    for i in range(len(branch_resistance)):
        r[i][i] = branch_resistance[i]

    # Initialize links
    links = np.zeros((len(branch_indexes), len(nodes_power)))

    # Create linkes between lines and nodes
    for i in range(len(branch_indexes)):
        links[i][branch_indexes[i][0]] = 1
        links[i][branch_indexes[i][1]] = -1

    # Remove first colomn and rename to A_t
    shapes = np.shape(links)
    A_t = np.zeros((shapes[0], shapes[1] - 1))
    for i in range(len(links)):
        for j in range(1, len(links[i])):
            A_t[i, j - 1] = links[i, j]

    # Solve matrices
    A = np.transpose(A_t)
    if len(branch_resistance) > 1:
        x_inv = np.linalg.inv(r)
    else:
        x_inv = 1 / r
    P = np.array(nodes_power[1:])

    B = A.dot(x_inv).dot(A_t)

    def fun(thetas):
        return B.dot(thetas) - P

    res = opt.fsolve(fun, np.zeros(len(P)))
    sol = np.concatenate(([0], res))
    sol_branch = []
    for i in range(len(r)):
        sol_branch.append(
            np.round(
                (1 / r[i][i]) * (sol[branch_indexes[i][0]] - sol[branch_indexes[i][1]]),
                decimals=2,
            )
        )

    return sol_branch
