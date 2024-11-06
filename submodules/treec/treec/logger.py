import shutil
import timeit
from treec.visualise_tree import binarytree_to_dot
from treec.tree import BinaryTreeFree
import os
import json
from filelock import FileLock


class GeneralLogger:
    def __init__(self, save_dir, algo_type, common_params, algo_params):
        self.save_dir = save_dir
        self.algo_type = algo_type
        self.common_params = common_params
        self.algo_params = algo_params

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.folder_name = self.create_dir()

        self.initialise_dir()

        self.model_dir = self.folder_name + "models/"
        os.mkdir(self.model_dir)

        self.power_prof_dir = self.folder_name + "power_profiles/"
        os.mkdir(self.power_prof_dir)

        self.rewards_dir = self.folder_name + "rewards/"
        os.mkdir(self.rewards_dir)

        self.attributes_dir = self.folder_name + "attributes/"
        os.mkdir(self.attributes_dir)

        self.eval_score_path = self.folder_name + "eval_score.csv"

        file = open(self.eval_score_path, "w+")
        file.write("timestep,elapsed_time,eval_score\n")
        file.close()

        self.episode_score_file = self.folder_name + "episode_score_file.csv"
        self.create_episode_score_file()

        self.start_time = timeit.default_timer()

    def create_dir(self):
        config_file = self.common_params["config_file"]

        microgrid_name = config_file.split("/")[-1].split(".")[0]
        folder_start = microgrid_name + "_" + self.algo_type + "_"

        subfolders = [name for name in os.listdir(self.save_dir)]
        highest_num = -1
        for folder in subfolders:
            if folder.startswith(folder_start):
                folder_num = int(folder.replace(folder_start, ""))
                if folder_num > highest_num:
                    highest_num = folder_num

        folder_name = self.save_dir + "/" + folder_start + str(highest_num + 1) + "/"

        os.mkdir(folder_name)
        return folder_name

    def initialise_dir(self):
        config_file = self.common_params["config_file"]
        logged_config_file = self.folder_name + "microgrid_config.csv"

        shutil.copyfile(config_file, logged_config_file)

        params_str = ""
        params_str += self.algo_type + "\n\n"
        params_str += "common params:" + "\n"
        for key, value in self.common_params.items():
            if callable(value):
                val_str = value.__name__
            else:
                val_str = str(value)
            params_str += key + "," + val_str + "\n"

        params_str += "\nalgo params:\n"
        for key, value in self.algo_params.items():
            if callable(value):
                val_str = value.__name__
            else:
                val_str = str(value)
            params_str += key + "," + val_str + "\n"

        param_filepath = self.folder_name + "params_run.csv"

        text_file = open(param_filepath, "w+")
        text_file.write(params_str)
        text_file.close()

    def episode_eval_log(self, model, eval_score):
        with FileLock(self.episode_score_file + str(".lock"), mode=0o664):
            episode_count, _ = self.read_best_score_episode_count()
            self.update_episode_count()
            better_score = self.update_best_score(eval_score)

        score_str = "{0:.1f}".format(eval_score)

        eval_exten = str(episode_count) + "_" + score_str

        time_step = str(episode_count * self.common_params["tot_steps"])

        elapsed_time = str(timeit.default_timer() - self.start_time)

        text_file = open(self.eval_score_path, "a+")
        text_file.write(time_step + "," + elapsed_time + "," + score_str + "\n")
        text_file.close()

        if better_score:
            return eval_exten

        return None

    def save_model(self, model, eval_exten):
        pass

    def create_episode_score_file(self):
        file = open(self.episode_score_file, "w+")
        file.write("episode,0\nscore,\n")
        file.close()

    def update_episode_count(self):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            episode_count = int(file_content[0].replace("episode,", ""))
        except (ValueError, IndexError):
            print(file_content)
            return

        new_episode_count = episode_count + 1

        file_content[0] = f"episode,{new_episode_count}"

        file = open(self.episode_score_file, "w+")
        file.write("\n".join(file_content))
        file.close()

    def update_best_score(self, new_best_score):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            best_score = file_content[1].replace("score,", "")
        except IndexError:
            print(file_content)
            return False

        if best_score == "" or float(best_score) < new_best_score:
            file_content[1] = f"score,{new_best_score}"

            file = open(self.episode_score_file, "w+")
            file.write("\n".join(file_content))
            file.close()

            return True

        return False

    def read_best_score_episode_count(self):
        file = open(self.episode_score_file, "r+")
        file_content = file.read().split("\n")
        file.close()
        try:
            episode_count = int(file_content[0].replace("episode,", ""))
        except (ValueError, IndexError):
            episode_count = 0
            print(file_content)

        try:
            if file_content[1].replace("score,", "") == "":
                best_score = None
            else:
                best_score = float(file_content[1].replace("score,", ""))
        except (ValueError, IndexError):
            best_score = 100
            print(file_content)

        return episode_count, best_score


class TreeLogger(GeneralLogger):
    def __init__(self, save_dir, algo_type, common_params, algo_params):
        super().__init__(save_dir, algo_type, common_params, algo_params)

        self.dot_files_dir = self.folder_name + "dot_trees/"
        os.mkdir(self.dot_files_dir)

    def save_model(self, model, eval_exten):
        json_model = dict()
        for key, value in model.items():
            if key != "trees":
                json_model[key] = value

        for i, tree in enumerate(model["trees"]):
            tree_model = tree.get_dict_model()
            json_model[f"tree_{i}"] = tree_model

        model_str = json.dumps(json_model)

        model_path = self.model_dir + "model_" + eval_exten + ".json"

        file = open(model_path, "w+")

        file.write(model_str)

        file.close()

    def save_tree_dot(self, trees, all_nodes_visited, eval_exten):
        for i, tree in enumerate(trees):
            leafs_batt = [j[i] for j in all_nodes_visited]
            title = "Tree_" + str(i)
            dot_str = binarytree_to_dot(tree, title, leafs_batt)
            file = open(self.dot_files_dir + eval_exten + "_" + title + ".dot", "w+")
            file.write(dot_str)
            file.close()

    @staticmethod
    def get_best_model(model_folder):
        model_files = [
            f
            for f in os.listdir(model_folder)
            if os.path.isfile(os.path.join(model_folder, f))
        ]

        best_model = model_files[0]

        for model_file in model_files:
            best_score = float(best_model.split("_")[-1].replace(".json", ""))
            model_score = float(model_file.split("_")[-1].replace(".json", ""))

            best_episode = int(best_model.split("_")[-2].replace(".json", ""))
            model_episode = int(model_file.split("_")[-2].replace(".json", ""))

            if model_score >= best_score and model_episode > best_episode:
                best_model = model_file
        best_model_file = model_folder + best_model

        with open(best_model_file) as file:
            model = json.load(file)

        return model

    @staticmethod
    def get_best_trees(model_folder):
        tree_model = TreeLogger.get_best_model(model_folder)
        trees = list()

        trees_num = [
            int(key.replace("tree_", ""))
            for key in tree_model.keys()
            if key.startswith("tree_")
        ]

        for tree_num in sorted(trees_num):
            tree_parameters = tree_model[f"tree_{tree_num}"]
            node_array = tree_parameters["node_array"]
            feature_info = tree_parameters["feature_info"]
            action_names = tree_parameters["action_names"]
            tree = BinaryTreeFree(node_array, feature_info, action_names)
            trees.append(tree)

        return trees
