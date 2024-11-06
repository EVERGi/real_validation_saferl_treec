import datetime
import pickle
import numpy as np
from simugrid.management.rational import RationalManager
from simugrid.simulation.config_parser import parse_config_file

import sys
import os
import math

filed_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(filed_dir)
sys.path.append(parent_dir)

from reproduce_svl.custom_classes import DayAheadEngie
import matplotlib.pyplot as plt


class ModelMicroForecaster:
    def __init__(self, mode, house_num):
        self.mode = mode
        self.car_cap = 60
        self.cur_det = 0
        self.pred_det = 0
        self.pred_soc_f = 0
        self.max_charge = 7.4
        self.house_num = house_num
        self.switch_hour = None
        self.forec_dt = None

    def pv_model_forecast(self, microgrid, solar_pv):
        end_dt = microgrid.utc_datetime
        start_dt = end_dt - datetime.timedelta(days=1)

        pv_lag = solar_pv.get_forecast(start_dt, end_dt, quality="perfect")[
            "electric_power"
        ]["values"][::-1]
        hour = microgrid.utc_datetime.hour

        input_data = pv_lag + [hour]
        input_data = np.array(input_data).reshape(1, -1)
        predicted_pvs = list(self.pv_model.predict(input_data)[0])

        predicted_pvs = [i if i > 0 else 0 for i in predicted_pvs]

        return predicted_pvs

    def load_model_forecast(self, microgrid, consumer):
        end_dt = microgrid.utc_datetime
        start_dt = end_dt - datetime.timedelta(days=1)

        load_lag = consumer.get_forecast(start_dt, end_dt, quality="perfect")[
            "electric_power"
        ]["values"][::-1]
        hour = microgrid.utc_datetime.hour
        day_of_week = microgrid.utc_datetime.weekday()

        input_data = [hour] + load_lag + [day_of_week]
        input_data = np.array(input_data).reshape(1, -1)
        predicted_loads = list(self.load_model.predict(input_data)[0])

        predicted_loads = [i if i > 0 else 0 for i in predicted_loads]

        return predicted_loads

    def ev_model_forecast(self, microgrid, soc_i_env):
        hour = microgrid.utc_datetime.hour

        det = self.ev_model_detention.predict(
            np.array([soc_i_env, hour]).reshape(1, -1)
        )[0]

        if self.switch_hour is not None:
            det = self.correct_det_switch_time(microgrid, det)
        soc_f = self.ev_model_soc.predict(np.array([soc_i_env, hour]).reshape(1, -1))[0]

        return det, soc_f

    def correct_det_switch_time(self, microgrid, det_h):
        next_switch_time = microgrid.utc_datetime.replace(
            hour=self.switch_hour, minute=0, second=0
        )
        if microgrid.utc_datetime > next_switch_time:
            next_switch_time += datetime.timedelta(days=1)
        time_to_switch = next_switch_time - microgrid.utc_datetime
        det_time_delta = datetime.timedelta(hours=det_h)
        if det_time_delta > time_to_switch:
            det_h = time_to_switch.seconds / 3600

        return det_h

    def set_hour_day_switch(self, switch_hour):
        self.switch_hour = switch_hour

    def forecast_pv_load(self, microgrid):
        for asset in microgrid.assets:
            if asset.name == "SolarPv_0":
                solar_pv = asset
        start_t = microgrid.utc_datetime
        end_t = start_t + datetime.timedelta(days=1)

        if self.mode in ["naive", "perfect"]:
            pv_load = solar_pv.get_forecast(start_t, end_t, quality=self.mode)[
                "electric_power"
            ]["values"]
        else:
            pv_load = self.pv_model_forecast(microgrid, solar_pv)

        return pv_load

    def forecast_consum_load(self, microgrid):
        for asset in microgrid.assets:
            if asset.name == "Consumer_0":
                consumer = asset
        start_t = microgrid.utc_datetime
        end_t = start_t + datetime.timedelta(days=1)

        if self.mode in ["naive", "perfect"]:
            consumer_load = consumer.get_forecast(
                start_t,
                end_t,
                quality=self.mode,
                naive_back=datetime.timedelta(days=1),
            )["electric_power"]["values"]
        else:
            consumer_load = self.load_model_forecast(microgrid, consumer)

        consumer_load = [-i for i in consumer_load]
        return consumer_load

    def forecast_charger(self, microgrid):
        for asset in microgrid.assets:
            if asset.name == "Charger_0":
                charger = asset
        start_t = microgrid.utc_datetime
        end_t = start_t + datetime.timedelta(days=1)

        end_one_step = start_t + microgrid.time_step
        env_values = microgrid.environments[0].env_values
        soc_i_env = env_values["soc_i_0"].get_forecast(start_t, end_one_step)["values"][
            0
        ]

        if self.mode == "perfect":
            det_vals = env_values["det_0"].get_forecast(start_t, end_t)["values"]
            cap_vals = env_values["capa_0"].get_forecast(start_t, end_t)["values"]
            soc_i_vals = env_values["soc_i_0"].get_forecast(start_t, end_t)["values"]
            soc_f_vals = env_values["soc_f_0"].get_forecast(start_t, end_t)["values"]
            p_max_vals = env_values["p_max_0"].get_forecast(start_t, end_t)["values"]

            # Added buffer time to perfect forecaster
            MAX_BUFFER_SEC = 4 * 60 * 60
            MIN_BUFFER_SEC = 3 * 60
            soc_left = charger.soc_f - charger.soc
            buffer_seconds = (
                soc_left * (MAX_BUFFER_SEC - MIN_BUFFER_SEC) + MIN_BUFFER_SEC
            )
            buffer_h = buffer_seconds / 3600
            ts_h = microgrid.time_step.total_seconds() / 3600

            buffer_ts = buffer_h / ts_h
            if charger.det > 2:
                self.cur_det = charger.det - round(buffer_ts)
            else:
                self.cur_det = charger.det

            self.cur_soc_f = charger.soc_f

        else:
            det_vals = [0] * 24 * 4
            cap_vals = [0] * 24 * 4
            soc_i_vals = [0] * 24 * 4
            soc_f_vals = [0] * 24 * 4
            p_max_vals = [0] * 24 * 4

            if soc_i_env != 0:
                if self.mode == "model":
                    det, soc_f = self.ev_model_forecast(microgrid, soc_i_env)
                    det = int(det * 4)
                elif self.mode == "naive":
                    # Median detention time
                    det = int(4.25 * 4)
                    # Median soc final
                    soc_f = 0.94
                self.pred_det = det
                self.cur_det = det
                self.cur_soc_f = soc_f
                self.forec_dt = microgrid.utc_datetime
            elif self.forec_dt is not None:
                time_to_forec = microgrid.utc_datetime - self.forec_dt
                det = self.pred_det - time_to_forec.seconds // 900
                self.cur_det = det
                if det <= 0:
                    self.cur_soc_f = 0

        if charger.det > 0 and self.cur_det > 0:
            det_vals[0] = self.cur_det
            soc_f_vals[0] = self.cur_soc_f

            cap_vals[0] = self.car_cap
            soc_i_vals[0] = charger.soc

            p_max_vals[0] = self.max_charge

        charge_forec = {
            "det": det_vals,
            "cap": cap_vals,
            "soc_i": soc_i_vals,
            "soc_f": soc_f_vals,
            "p_max": p_max_vals,
        }
        return charge_forec

    def forecast_day_ahead(self, microgrid):
        start_t = microgrid.utc_datetime
        end_t = start_t + datetime.timedelta(days=1)

        day_ahead_val = microgrid.environments[0].env_values["day_ahead_price"]
        day_ahead = day_ahead_val.get_forecast(start_t, end_t)["values"]

        return day_ahead


class ThreeMonthForecaster(ModelMicroForecaster):
    def __init__(
        self,
        mode,
        house_num,
        models_dir="data/ems_models/forecasting/",
    ):
        super().__init__(mode, house_num)
        if self.mode == "model":
            exten = "2024-01-01_0000_2024-04-01_0000"

            pv_model_file = f"{models_dir}pv_model_{exten}_{self.house_num}.pkl"
            load_model_file = f"{models_dir}load_model_{exten}.pkl"
            ev_model_file_detention = f"{models_dir}ev_model_detention_{exten}.pkl"
            ev_model_file_soc = f"{models_dir}ev_model_soc_{exten}.pkl"
            with open(pv_model_file, "rb") as f:
                self.pv_model = pickle.load(f)
            with open(load_model_file, "rb") as f:
                self.load_model = pickle.load(f)
            with open(ev_model_file_detention, "rb") as f:
                self.ev_model_detention = pickle.load(f)
            with open(ev_model_file_soc, "rb") as f:
                self.ev_model_soc = pickle.load(f)


if __name__ == "__main__":
    house_num = 5
    date_folder = "2024-01-01_0000_2024-04-01_0000"
    # config_file = f"data/houses/house_{house_num}/2023-09-08_10:00:00_2023-10-27_16:00:00/house_training_simul.json"
    config_file = f"data/houses/house_{house_num}/{date_folder}/house_train.json"
    microgrid = parse_config_file(config_file)
    reward = DayAheadEngie()
    microgrid.set_reward(reward)

    forec = ThreeMonthForecaster("model", house_num)

    RationalManager(microgrid)

    for i in range(300):
        microgrid.management_system.simulate_step()

    for assets in microgrid.assets:
        if assets.name == "Battery_0":
            battery = assets
        elif assets.name == "SolarPv_0":
            solar_pv = assets
        elif assets.name == "Consumer_0":
            consumer = assets
        elif assets.name == "Charger_0":
            charger = assets
    pv_load = forec.pv_model_forecast(microgrid, solar_pv)

    consum_load = forec.load_model_forecast(microgrid, consumer)
    print(consum_load)

    plt.figure()
    plt.plot(pv_load, label="pv")
    plt.plot(consum_load, label="consum")
    plt.legend()
    plt.show()


class BugForecaster(ThreeMonthForecaster):
    def pv_model_forecast(self, microgrid, solar_pv):
        predicted_pvs = super().pv_model_forecast(microgrid, solar_pv)

        start_bug_dt = datetime.datetime(2024, 5, 30, 15, 0, 0)
        end_bug_dt = datetime.datetime(2024, 5, 31, 6, 30, 0)

        cur_time = microgrid.utc_datetime.replace(tzinfo=None)
        if start_bug_dt <= cur_time < end_bug_dt:
            end_dt = microgrid.utc_datetime
            start_dt = end_dt - datetime.timedelta(days=1)
            pv_past = solar_pv.get_forecast(start_dt, end_dt, quality="perfect")[
                "electric_power"
            ]["values"]
            diff_end_bug = int((end_bug_dt - cur_time) / microgrid.time_step)
            pv_past_bug = pv_past[diff_end_bug:] + pv_past[:diff_end_bug]
            pv_lag = pv_past_bug[::-1]
            hour = microgrid.utc_datetime.hour

            input_data = pv_lag + [hour]
            input_data = np.array(input_data).reshape(1, -1)
            predicted_pvs = list(self.pv_model.predict(input_data)[0])

            predicted_pvs = [i if i > 0 else 0 for i in predicted_pvs]
        return predicted_pvs

    def load_model_forecast(self, microgrid, consumer):
        predicted_loads = super().load_model_forecast(microgrid, consumer)

        start_bug_dt = datetime.datetime(2024, 5, 30, 15, 0, 0)
        end_bug_dt = datetime.datetime(2024, 5, 31, 6, 45, 0)
        cur_time = microgrid.utc_datetime.replace(tzinfo=None)
        if start_bug_dt <= cur_time < end_bug_dt:

            end_dt = microgrid.utc_datetime
            start_dt = end_dt - datetime.timedelta(days=1)

            load_past = consumer.get_forecast(start_dt, end_dt, quality="perfect")[
                "electric_power"
            ]["values"]

            diff_end_bug = int((end_bug_dt - cur_time) / microgrid.time_step)
            load_past_bug = load_past[diff_end_bug:] + load_past[:diff_end_bug]

            load_lag = load_past_bug[::-1]

            hour = microgrid.utc_datetime.hour
            day_of_week = microgrid.utc_datetime.weekday()

            input_data = [hour] + load_lag + [day_of_week]
            input_data = np.array(input_data).reshape(1, -1)
            predicted_loads = list(self.load_model.predict(input_data)[0])

            predicted_loads = [i if i > 0 else 0 for i in predicted_loads]

        return predicted_loads

    def forecast_charger(self, microgrid):
        charge_forec = super().forecast_charger(microgrid)

        cur_time = microgrid.utc_datetime
        if cur_time.hour == 15 and cur_time.minute == 0:
            charge_forec = {
                "det": [0] * 24 * 4,
                "cap": [0] * 24 * 4,
                "soc_i": [0] * 24 * 4,
                "soc_f": [0] * 24 * 4,
                "p_max": [0] * 24 * 4,
            }
        return charge_forec
