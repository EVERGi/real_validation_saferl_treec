import csv
from simugrid.simulation import Microgrid, Node, Environment, Branch
import datetime
from simugrid.utils import ROOT_DIR
import sys

from simugrid.assets import __all__ as all_simu_assets
from simugrid.assets import *

import os
import json


def strisfloat(string):
    if not isinstance(string, str):
        return False
    try:
        float(string)
        return True
    except (ValueError, TypeError):
        return False


def strisint(string):
    if not isinstance(string, str):
        return False
    try:
        int(string)
        return True
    except (ValueError, TypeError):
        return False


def import_all_assets(custom_class):
    asset_class_dict = dict()

    for asset_class in all_simu_assets:
        asset = getattr(sys.modules[__name__], asset_class)
        if asset_class not in custom_class.keys():
            asset_class_dict[asset_class] = asset
        else:
            asset_class_dict[asset_class] = custom_class[asset_class]

    ignore_objects = ["Microgrid", "Node", "object", "AssetM"]
    for custom_asset_name, custom_asset_class in custom_class.items():
        if custom_asset_name in ignore_objects:
            continue
        if custom_asset_name not in asset_class_dict.keys():
            asset_class_dict[custom_asset_name] = custom_asset_class

    return asset_class_dict


def parse_microgrid(rows, microgrid, custom_class):
    dict_args = {row[0]: row[1] for row in rows}

    # Keys names change for json config files
    # To delete once csv are not used anymore
    start_time = "start_time"
    end_time = "end_time"
    time_step = "time_step"
    number_of_nodes = "number_of_nodes"
    if start_time not in dict_args:
        start_time = "start_time"
        end_time = "end_time"
        time_step = "time_step"
        number_of_nodes = "number_of_nodes"

    formats = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]
    for form in formats:
        try:
            start_t = datetime.datetime.strptime(dict_args[start_time], form)
            end_t = datetime.datetime.strptime(dict_args[end_time], form)
            break
        except ValueError as e:
            if form == formats[-1]:
                raise e

    datetime_step = datetime.datetime.strptime(dict_args[time_step], "%H:%M:%S")
    time_step = datetime.timedelta(
        hours=datetime_step.hour,
        minutes=datetime_step.minute,
        seconds=datetime_step.second,
    )

    if "Microgrid" not in custom_class.keys():
        microgrid = Microgrid(start_t, end_t, time_step, timezone=dict_args["timezone"])
    else:
        microgrid = custom_class["Microgrid"](
            start_t, end_t, time_step, timezone=dict_args["timezone"]
        )

    for _ in range(int(dict_args[number_of_nodes])):
        if "Node" not in custom_class.keys():
            Node(microgrid)
        else:
            custom_class["Node"](microgrid)

    return microgrid


def parse_branch(dict_args, microgrid, custom_class):
    if "nodes_index" not in dict_args.keys():
        return microgrid

    nodes_index = [int(i) for i in dict_args["nodes_index"].split("to")]

    if "Branch" not in custom_class.keys():
        branch = Branch(
            nodes_index,
        )
    else:
        branch = custom_class["Branch"](
            nodes_index,
        )
    attributes = attribute_list_to_dict(dict_args)
    branch.set_attributes(attributes)
    microgrid.branches.append(branch)
    return microgrid


def parse_branches(rows, microgrid, custom_class):
    branch_rows = []
    for row in rows:
        if row[0].replace("Branch ", "").isnumeric():
            dict_args = {row[0]: row[1] for row in branch_rows}
            parse_branch(dict_args, microgrid, custom_class)
            branch_rows = []
        else:
            branch_rows += [row]

    dict_args = {row[0]: row[1] for row in branch_rows}
    parse_branch(dict_args, microgrid, custom_class)

    return microgrid


def attribute_list_to_dict(dict_args, node_number="node_number"):
    attributes = {
        key: None if value == "" else value
        for key, value in dict_args.items()
        if key not in ["name", node_number, "nodes_index"]
    }

    attributes = {
        key: int(value) if strisint(value) else value
        for key, value in attributes.items()
    }
    attributes = {
        key: float(value) if strisfloat(value) and isinstance(value, str) else value
        for key, value in attributes.items()
    }
    attributes = {
        key: str2bool(value) if isinstance(value, str) and isBool(value) else value
        for key, value in attributes.items()
    }
    attributes.pop("", None)

    return attributes


def str2bool(v):
    return v.lower() == "true"


def isBool(v):
    return v.lower() in ["true", "false"]


def parse_asset(dict_args, microgrid, custom_class):
    if "name" not in dict_args.keys():
        return

    # Keys names change for json config files
    # To delete once csv are not used anymore
    node_number = "node_number"
    if node_number not in dict_args:
        node_number = "node_number"

    class_name = dict_args["name"].split("_")[0]

    classes = import_all_assets(custom_class)
    asset_class = classes[class_name]

    node_num = int(dict_args[node_number])
    asset = asset_class(microgrid.nodes[node_num], dict_args["name"])

    attributes = attribute_list_to_dict(dict_args, node_number)
    asset.set_attributes(attributes)


def parse_assets(rows, microgrid, custom_class):
    asset_rows = []
    for row in rows:
        if row[0].replace("Asset_", "").isnumeric():
            dict_args = {row[0]: row[1] for row in asset_rows}
            parse_asset(dict_args, microgrid, custom_class)
            asset_rows = []
        else:
            asset_rows += [row]
    dict_args = {row[0]: row[1] for row in asset_rows}
    parse_asset(dict_args, microgrid, custom_class)
    return microgrid


def parse_environment(env_rows, microgrid, config_file):
    if type(config_file) in [list, dict]:
        point_dir = os.getcwd()
    else:
        abs_path = os.path.abspath(config_file)
        abs_path = abs_path.replace("\\", "/")
        point_dir = "/".join(abs_path.split("/")[:-1])

    new_env = Environment(microgrid)
    env_nodes = []
    for row in env_rows:
        if row[0] == "nodes_number":
            if isinstance(row[1], str):
                node_nums = row[1].split(",")
            else:
                node_nums = row[1]
            env_nodes = [int(node) for node in node_nums]

        elif row[0][-3:] == ".py":
            if row[0][0] == "/" or row[0][1:3] == ":/":
                new_env.add_function_values(row[0])
            elif row[0][0] == ".":
                new_env.add_function_values(point_dir + row[0][1:])
            else:
                new_env.add_function_values(ROOT_DIR + row[0])
        elif row[0][-4:] == ".csv":
            if row[0][0] == "/" or row[0][1:3] == ":/":
                new_env.add_multicolumn_csv_values(row[0])
            elif row[0][0] == ".":
                new_env.add_multicolumn_csv_values(point_dir + row[0][1:])
            else:
                new_env.add_multicolumn_csv_values(ROOT_DIR + row[0])
        elif strisfloat(row[1]):
            new_env.add_value(row[0], float(row[1]))
        elif not isinstance(row[1], str):
            new_env.add_value(row[0], row[1])
        elif row[1][-4:] == ".csv":
            if row[1][0] == "/" or row[1][1:3] == ":/":
                new_env.add_value(row[0], row[1])
            elif row[1][0] == ".":
                new_env.add_value(row[0], point_dir + row[1][1:])
            else:
                new_env.add_value(row[0], ROOT_DIR + row[1])

    for node_ind in env_nodes:
        microgrid.nodes[node_ind].set_environment(new_env)


def parse_environments(rows, microgrid, config_file):
    environment_rows = None
    for row in rows:
        if row[0].replace("Environment_", "").isnumeric() and environment_rows is None:
            environment_rows = []
        elif environment_rows is None:
            continue
        elif row[0].replace("Environment_", "").isnumeric():
            environment_rows = []
            parse_environment(environment_rows, microgrid, config_file)
        else:
            environment_rows += [row]

    parse_environment(environment_rows, microgrid, config_file)

    return microgrid


def set_model_all_assets(microgrid):
    for node in microgrid.nodes:
        for asset in node.assets:
            asset.check_and_set_model()


def parse_config_file(config_file, custom_class=dict()):
    """
    :ivar config_file: the path to config_file
    :type config_file: string
    :ivar custom_class: the custom classes for microgrid, node, assets
    :type custom_class: dict

    :return: the parsed microgrid
    :rtype: Microgrid
    """
    current_section = ""
    row_saver = []

    microgrid = None

    sections = ["Microgrid", "Branches", "Assets", "Environments", "END"]

    if type(config_file) == list:
        config_list = config_file
    elif type(config_file) == dict:
        config_list = config_dict_to_config_list(config_file)
    elif config_file.endswith(".csv"):
        with open(config_file) as csvfile:
            csvreader = csv.reader(csvfile)
            config_list = list(csvreader)
    elif config_file.endswith(".json"):
        with open(config_file) as jsonfile:
            config_dict = json.load(jsonfile)
        config_list = config_dict_to_config_list(config_dict)

    else:
        raise Exception("Config file is not a string path neither a list.")

    for row in config_list:
        new_section = False
        if len(row) == 0:
            continue
        if row[0] in sections:
            new_current_section = row[0]
            new_section = True

        if new_section:
            if current_section == "Microgrid":
                microgrid = parse_microgrid(row_saver, microgrid, custom_class)
            elif current_section == "Branches":
                microgrid = parse_branches(row_saver, microgrid, custom_class)
            elif current_section == "Assets":
                microgrid = parse_assets(row_saver, microgrid, custom_class)
            elif current_section == "Environments":
                microgrid = parse_environments(row_saver, microgrid, config_file)
            current_section = new_current_section
            row_saver = []
        else:
            row_saver += [row]

    set_model_all_assets(microgrid)

    microgrid.env_simulate()

    return microgrid


def config_dict_to_config_list(config_dict):
    config_list = []
    recur_dict_to_config_list(config_dict, config_list)
    config_list += [["END"]]
    return config_list


def recur_dict_to_config_list(config_dict, config_list):
    for key, val in config_dict.items():
        if type(val) == dict:
            config_list += [[key]]
            recur_dict_to_config_list(val, config_list)
            config_list += [[]]
        else:
            config_list += [[key, val]]


if __name__ == "__main__":
    simple_config = {
        "Microgrid": {
            "number_of_nodes": "1",
            "start_time": "01/01/2000 00:00:00",
            "end_time": "07/01/2000 23:00:00",
            "timezone": "UTC",
            "time_step": "01:00:00",
        },
        "Assets": {
            "Asset_0": {
                "node_number": "0",
                "name": "WindTurbine_0",
                "v_cin": "4",
                "v_cout": "25",
                "v_rated": "10",
                "size": "500",
            },
            "Asset_1": {
                "node_number": "0",
                "name": "PublicGrid_0",
            },
        },
        "Environments": {"Environment_0": {"nodes_number": "0", "wind_speed": "5"}},
    }
    config_list = config_dict_to_config_list(simple_config)

    for row in config_list:
        print(row)

    m = parse_config_file(simple_config)
    print(m)
