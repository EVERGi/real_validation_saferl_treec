import os
import sys

from matplotlib import pyplot as plt
from simugrid.misc.log_plot_micro import plot_attributes, plot_hist
from simugrid.simulation.config_parser import parse_config_file


filed_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(filed_dir)
sys.path.append(parent_dir)

from reproduce_svl.custom_classes import RationalSVL, SVLBattCharg, SVLTreeBattCharg
from reproduce_svl.mpc import SVLMPCManager, SimulExpMPC
from reproduce_svl.reproduce_experiment import (
    ExperimentReward,
    ThreeMonthsTree,
    calc_real_costs,
    log_results,
)

import gymnasium as gym
import numpy as np
from typing import TypeVar

from stable_baselines3 import TD3
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecCheckNan,
    VecMonitor,
)

from stable_baselines3.common.noise import NormalActionNoise


from gymnasium import spaces
from gymnasium.envs.registration import register


SelfOffPolicyAlgorithm = TypeVar("SelfOffPolicyAlgorithm", bound="OffPolicyAlgorithm")
SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class ExecRL(SVLBattCharg):
    def __init__(self, microgrid, params_evaluation=None):
        self.action = params_evaluation["action"]
        super().__init__(microgrid)

    def get_actions(self):
        return self.action, None


class HouseEnv(gym.Env):
    def __init__(
        self, house_num, date_range, env_to_ems, log_folder=None, use_cons_data=False
    ):
        self.house_num = house_num
        self.date_range = date_range
        self.env_to_ems = env_to_ems
        self.use_cons_data = use_cons_data
        self.log_folder = log_folder

        self.step_counter = 0
        self.already_done = False

        # state = np.array(
        #    [
        #        power_load,
        #        power_solar,
        #        soc_bess,
        #        soc_ev,
        #        power_price,
        #        hour_of_day,
        #        day_of_week,
        #    ],
        #    dtype=np.float32,
        # )

        # TODO find size of observation space
        obs_high = np.full(7, np.Inf)  # set no limit on observation space

        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32
        )  # normalised action space
        print(f"Gymnasium init for house")

        self.init_environment()

    def init_environment(self):
        switch_hour = 15

        config_file = f"data/houses/house_{self.house_num}/{self.date_range}/house_batt_not_fixed.json"

        if self.use_cons_data:
            # Read config_file and replace "./environment/cons.csv" with "./../../common_env/SFH19_2023_2024_15min_original.csv"
            with open(config_file, "r") as f:
                file_content = f.read()
            file_content = file_content.replace(
                "./environment/cons.csv",
                "./../../common_env/SFH19_2023_2024_15min_original.csv",
            )
            original_cons_conf = config_file.replace(
                "house_batt_not_fixed", "house_original_cons"
            )
            with open(original_cons_conf, "w+") as f:
                f.write(file_content)
            config_file = original_cons_conf

        microgrid = parse_config_file(config_file)
        self.microgrid = microgrid
        reward = ExperimentReward()
        microgrid.set_reward(reward)

        env_keys_to_log = [
            "day_ahead_price",
            "soc_i_0",
            "soc_f_0",
            "p_max_0",
            "capa_0",
            "PublicGrid_0_export",
            "PublicGrid_0_import",
            "ems_used",
            "success",
        ]
        for env_key in env_keys_to_log:
            microgrid.env_to_log(env_key)

        microgrid.attribute_to_log("Battery_0", "soc")
        # count = 0

        ems_used_value = microgrid.environments[0].env_values["ems_used"]
        success = microgrid.environments[0].env_values["success"].value
        if success:
            EmsClass = self.env_to_ems[ems_used_value.value]
        else:
            EmsClass = self.env_to_ems["RBC"]

        if issubclass(EmsClass, SVLMPCManager):
            params_house = {
                "house_num": self.house_num,
                "ems_set": ems_used_value.value,
            }
            ems = EmsClass(microgrid, params_evaluation=params_house)
        elif issubclass(EmsClass, SVLTreeBattCharg):
            ems = EmsClass(microgrid, self.house_num)
        elif issubclass(EmsClass, ExecRL):
            action = [0, 0]
            params_house = {"action": action}
            ems = EmsClass(microgrid, params_house)
        else:
            ems = EmsClass(microgrid)
        ems.set_hour_day_switch(switch_hour)
        self.prev_ems_used = ems_used_value.value

        while microgrid.datetime != microgrid.end_time:
            success = microgrid.environments[0].env_values["success"].value
            if success:
                NewEmsClass = self.env_to_ems[ems_used_value.value]
            else:
                NewEmsClass = self.env_to_ems["RBC"]

            ems_switch = ems_used_value.value != self.prev_ems_used
            if issubclass(NewEmsClass, ExecRL):
                break

            if ems_switch:
                self.prev_ems_used = ems_used_value.value

                EmsClass = NewEmsClass
                if issubclass(EmsClass, SVLMPCManager):
                    params_house = {
                        "house_num": self.house_num,
                        "ems_set": ems_used_value.value,
                    }
                    ems = EmsClass(microgrid, params_evaluation=params_house)
                elif issubclass(EmsClass, SVLTreeBattCharg):
                    ems = EmsClass(microgrid, self.house_num)
                else:
                    ems = EmsClass(microgrid)

                ems.set_hour_day_switch(switch_hour)
            # for _ in range(24 * 4):
            # count += 1
            # if count % 100 == 0:
            #    print(count)
            #    if count == 600:
            #        print(count)

            microgrid.management_system.simulate_step()

    def denormalise_actions(self, action):
        """
        Converts normalised actions (that have a value between -1 and +1 for the bess and 0 and +1 for the ev) into their actual denormalized values
        """
        for asset in self.microgrid.assets:
            if asset.name == "Battery_0":
                battery = asset
            if asset.name == "Charger_0":
                charger = asset

        min_battery = abs(battery.max_consumption_power)
        max_battery = abs(battery.max_production_power)
        min_charger = 0
        max_charger = abs(charger.max_consumption_power)

        _action = np.copy(action)
        if _action[0] >= 0:
            denormal_battery = _action[0] * max_battery
        else:
            denormal_battery = _action[0] * min_battery

        denormal_charger = _action[1] * max_charger
        denormalized_actions = [
            np.clip(denormal_battery, -min_battery, max_battery),
            np.clip(denormal_charger, max_charger, min_charger),
        ]

        return np.array(denormalized_actions)

    """
    def normalise_actions(self, action):
        pass
    """

    def get_observation(self):
        microgrid = self.microgrid

        for asset in microgrid.assets:
            if asset.name == "Battery_0":
                battery = asset
            if asset.name == "Charger_0":
                charger = asset

        load = microgrid.power_hist[0]["Consumer_0"][-1].electrical * 1000
        if len(microgrid.nodes) == 2:
            solar = microgrid.power_hist[1]["SolarPv_0"][-1].electrical * 1000
        else:
            solar = microgrid.power_hist[0]["SolarPv_0"][-1].electrical * 1000
        soc_bess = battery.soc * 100
        soc_ev = charger.soc * 100
        price = microgrid.environments[0].env_values["day_ahead_price"].value
        hour = microgrid.datetime.hour
        day = microgrid.datetime.weekday()

        obs = np.array(
            [
                load,
                solar,
                soc_bess,
                soc_ev,
                price,
                hour,
                day,
            ],
            dtype=np.float32,
        )

        return obs

    def normalise_observations(self, obs):
        # fmt: off
        min_load = 0; max_load = -9200 # warning: this can be higher
        min_solar = 0; max_solar = self.house.config_data["pv"]["max_power_kw"] * 1e3
        min_soc_bess = 0; max_soc_bess = 100
        min_soc_ev = 0; max_soc_ev = 100
        min_price = -0.120000; max_price = 0.330360 # warning: this is from 2023
        min_hour = 0; max_hour = 23
        min_day = 0; max_day = 6
        min_max_values = [(min_load, max_load), (min_solar, max_solar), 
                          (min_soc_bess, max_soc_bess), (min_soc_ev, max_soc_ev),
                          (min_price, max_price), (min_hour, max_hour),
                          (min_day, max_day)]
        # fmt: on

        normalized_obs = []
        denormal_obs = np.copy(np.float32(obs))
        for _obs, (min_val, max_val) in zip(denormal_obs, min_max_values):
            normalized = (2.0 * (_obs - min_val)) / (max_val - min_val) - 1.0
            normalized_obs.append(normalized)

        return np.array(normalized_obs, dtype=np.float32)

    def get_reward(self):
        grid_power = self.microgrid.power_hist[0]["PublicGrid_0"][-1].electrical

        day_ahead = self.microgrid.environments[0].env_values["day_ahead_price"].value
        offtake_extra = self.microgrid.environments[0].env_values["offtake_extra"].value
        injection_extra = (
            self.microgrid.environments[0].env_values["injection_extra"].value
        )
        kwh_offtake_cost = (
            self.microgrid.environments[0].env_values["kwh_offtake_cost"].value
        )

        if grid_power > 0:
            day_ahead = grid_power * (day_ahead + offtake_extra) * 1.06
            extra_offtake = kwh_offtake_cost * grid_power
        else:
            day_ahead = grid_power * (day_ahead + injection_extra)
            extra_offtake = 0

        reward = -day_ahead - extra_offtake
        reward *= 100  # Convert to euro cents

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        obs = np.array(
            [
                0,  # power_load,
                0,  # power_solar,
                0,  # soc_bess,
                0,  # soc_ev,
                0,  # power_price,
                0,  # hour_of_day,
                0,  # day_of_week,
            ],
            dtype=np.float32,
        )
        info = {}
        return obs, info

    def step(self, action):
        denorm_action = self.denormalise_actions(action)
        self.step_counter += 1

        switch_hour = 15
        microgrid = self.microgrid

        ems_used_value = microgrid.environments[0].env_values["ems_used"]

        while microgrid.datetime != microgrid.end_time:
            success = microgrid.environments[0].env_values["success"].value

            if success == "success":
                NewEmsClass = self.env_to_ems[ems_used_value.value]
            else:
                NewEmsClass = self.env_to_ems["RBC"]

            ems_switch = ems_used_value.value != self.prev_ems_used

            if issubclass(NewEmsClass, ExecRL):
                self.prev_ems_used = ems_used_value.value
                params_house = {"action": denorm_action}
                ems = NewEmsClass(microgrid, params_house)
                ems.set_hour_day_switch(switch_hour)

                ems.simulate_step()
                break
            if ems_switch:
                self.prev_ems_used = ems_used_value.value

                EmsClass = NewEmsClass
                if issubclass(EmsClass, SVLMPCManager):
                    params_house = {
                        "house_num": self.house_num,
                        "ems_set": ems_used_value.value,
                    }
                    ems = EmsClass(microgrid, params_evaluation=params_house)
                elif issubclass(EmsClass, SVLTreeBattCharg):
                    ems = EmsClass(microgrid, self.house_num)
                else:
                    ems = EmsClass(microgrid)

                ems.set_hour_day_switch(switch_hour)
            microgrid.management_system.simulate_step()

        rew = self.get_reward()
        done = False
        truncated = False
        obs = self.get_observation()

        experiment_done = not (microgrid.datetime != microgrid.end_time)

        done = experiment_done

        if done and not self.already_done:
            if self.log_folder is not None:
                log_results(microgrid, self.log_folder, self.house_num)
            calc_real_costs(microgrid)
            """
            power_hist = microgrid.power_hist
            kpi_hist = microgrid.reward_hist
            attributes_hist = microgrid.attributes_hist
            plot_hist(power_hist, kpi_hist)
            plot_attributes(attributes_hist)
            plt.tight_layout()
            plt.show()
            """
            self.already_done = True

        return obs, rew, done, truncated, {}


def run_rl_experiment(
    houses=[1, 2, 3, 5],
    log_folder=None,
    env_to_ems=None,
    use_cons_data=False,
    pretrain_dir="pre_training_rl/",
):
    date_vm = "2024-06-17_1500"
    date_range = f"2024-04-08_1500_{date_vm}"

    if env_to_ems is None:
        env_to_ems = {
            "RBC": RationalSVL,
            "MPC": SimulExpMPC,
            "RL": ExecRL,
            "Tree": ThreeMonthsTree,
        }

    for house_num in houses:
        register(id="HouseEnv-v0", entry_point=HouseEnv)
        kwargs = {
            "house_num": house_num,
            "date_range": date_range,
            "env_to_ems": env_to_ems,
            "log_folder": log_folder,
            "use_cons_data": use_cons_data,
        }
        env = VecCheckNan(
            VecMonitor(
                DummyVecEnv([lambda: gym.make("HouseEnv-v0", **kwargs)]),
            ),
            warn_once=False,
        )
        hyperp = {
            "gamma": 0.7,
            "learning_rate": 0.00058328,
            "batch_size": 16,
            "buffer_size": int(1e6),
            "train_freq": (4, "step"),  # ok that this is very low/frequent?
            "action_noise": NormalActionNoise(
                mean=np.zeros(env.action_space.shape[0]),
                sigma=0.182707 * np.ones(env.action_space.shape[0]),
            ),
            "learning_starts": 96,
        }

        agent = TD3("MlpPolicy", env, verbose=0, seed=0, **hyperp)

        agent = agent.load(
            f"{pretrain_dir}/td3_student_house{house_num}",
            env=env,
        )
        agent.load_replay_buffer(
            f"{pretrain_dir}/td3_student_house{house_num}_replay_buffer"
        )
        if house_num == 5:
            print("Executing agent for house 4")
        else:
            print(f"Executing agent for house {house_num}")
        print("This may take multiple minutes")
        agent.learn(
            total_timesteps=1153,
            reset_num_timesteps=False,
            tb_log_name=None,
        )


if __name__ == "__main__":
    log_folder = "results/experiment_1_gym/"
    run_rl_experiment(log_folder=log_folder)
