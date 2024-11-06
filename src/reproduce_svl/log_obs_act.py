import os
import sys
import datetime

filed_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(filed_dir)
sys.path.append(parent_dir)

from reproduce_svl.run_houses import run_house
from reproduce_svl.custom_classes import RationalSVL


def get_info_obs_act():
    house_data = {}
    for house_num in [1, 2, 3, 5]:
        date_folder = "2024-01-01_0000_2024-04-01_0000"
        config_file = f"data/houses/house_{house_num}/{date_folder}/house_train.json"
        Ems = RationalSVL
        microgrid = run_house(config_file, Ems, plot_graph=True, switch_hour=15)

        for asset in microgrid.assets:
            if asset.name == "Battery_0":
                battery = asset
            elif asset.name == "SolarPv_0":
                solar_pv = asset
            elif asset.name == "Consumer_0":
                consumer = asset
            elif asset.name == "Charger_0":
                charger = asset
        # History of battery soc and soc charger at the end of the timestep
        soc = microgrid.attributes_hist[battery]["soc"]
        soc_charger = microgrid.attributes_hist[charger]["soc"]

        # History of power load and power pv observation (power of previous 15 minutes)
        # The first element is the power of the second element because we don't have the power of 15 minutes before the simulation
        power_load = microgrid.power_hist[0][consumer.name]
        power_load = [power_load[0]] + power_load
        power_load = [p.electrical for p in power_load]
        if len(microgrid.nodes) == 2:
            power_pv = microgrid.power_hist[1][solar_pv.name]
        else:
            power_pv = microgrid.power_hist[0][solar_pv.name]
        power_pv = [power_pv[0]] + power_pv
        power_pv = [p.electrical for p in power_pv]

        start_dt = microgrid.start_time
        end_dt = microgrid.end_time

        # Get the datetime of each step
        datetime_hist = []
        cur_dt = start_dt
        while cur_dt < end_dt:
            datetime_hist.append(cur_dt)
            cur_dt += microgrid.time_step

        # Get hour and week day of each step
        hour_hist = [dt.hour for dt in datetime_hist]
        weekday_hist = [dt.weekday() for dt in datetime_hist]

        # Get power of battery and charger set for the next 15 minutes (action)
        if len(microgrid.nodes) == 2:
            power_battery = microgrid.power_hist[1][battery.name]
        else:
            power_battery = microgrid.power_hist[0][battery.name]

        power_battery = [p.electrical for p in power_battery]

        power_charger = microgrid.power_hist[0][charger.name]
        power_charger = [p.electrical for p in power_charger]

        # Get day ahead price at the timestep
        day_ahead = (
            microgrid.environments[0]
            .env_values["day_ahead_price"]
            .get_forecast(start_dt, end_dt)["values"]
        )

        # Get the soc init in the schedule, the soc history is the soc at the end of timestep
        # but when a new vehicle arrives at the same moment, the soc should be the arrival soc and not the one at the end of the previous timestep
        # so when there is an init soc (arrival soc) it will overwrite the end soc
        soc_init_charg = (
            microgrid.environments[0]
            .env_values["soc_i_0"]
            .get_forecast(start_dt, end_dt)["values"]
        )

        soc_charger[0] = soc_init_charg[0]

        # Overwrite the soc charger with the soc init when there is an init soc
        soc_charger = [
            (
                s
                if i >= len(soc_init_charg) or soc_init_charg[i] == 0
                else soc_init_charg[i]
            )
            for i, s in enumerate(soc_charger)
        ]

        # Format the data in a dict in the correct order
        data_dict = {
            "datetime": datetime_hist[: len(datetime_hist)],
            "power_load (kW)": power_load[: len(datetime_hist)],
            "power_solar (kW)": power_pv[: len(datetime_hist)],
            "soc_bess": soc[: len(datetime_hist)],
            "soc_ev": soc_charger[: len(datetime_hist)],
            "power_price (€/kWh)": day_ahead[: len(datetime_hist)],
            "hour_of_day": hour_hist[: len(datetime_hist)],
            "day_of_week": weekday_hist[: len(datetime_hist)],
            "power_battery (kW)": power_battery[: len(datetime_hist)],
            "power_charger (kW)": power_charger[: len(datetime_hist)],
        }
        house_data[house_num] = data_dict

    return house_data


def log_obs_act():
    house_data = get_info_obs_act()
    for house_num in [1, 2, 3, 5]:
        data_dict = house_data[house_num]
        with open(f"house_{house_num}_obs_act.csv", "w+") as f:
            data_keys = list(data_dict.keys())
            f.write(",".join(data_keys) + "\n")
            for i in range(len(data_dict["datetime"])):
                f.write(",".join([str(data_dict[k][i]) for k in data_keys]) + "\n")


if __name__ == "__main__":
    log_obs_act()
