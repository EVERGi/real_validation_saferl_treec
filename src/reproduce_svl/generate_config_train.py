import json

# Add python path to current folder
import sys
import os

filed_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filed_dir)

from imputation import impute_missing_values


def generate_config_train(data_folder):
    base_config = data_folder + "/house_batt_not_fixed.json"

    pv_file = data_folder + "/environment/solar.csv"
    output_pv = data_folder + "/environment/solar_imputed.csv"
    impute_missing_values(pv_file, output_pv)

    make_config_train(base_config)


def make_config_train(config_file):
    json_data = json.load(open(config_file))
    json_data["Assets"]["Asset 4"] = {
        "node number": 0,
        "name": "Charger_0",
        "max_charge_cp": 7.4,
        "max_discharge_cp": 0,
        "ID": 0,
    }
    environment = json_data["Environments"]["Environment 0"]

    # Add common environment values
    environment["Consumer_0_electric"] = "./../../common_env/SFH19_2018_2019_15min.csv"
    environment["./../../common_env/first_ev_schedule.csv"] = None

    # Remove unwanted environment values
    environment.pop("PublicGrid_0")
    environment.pop("soc_svl")
    environment["SolarPv_0"] = "./environment/solar_imputed.csv"

    new_config_file = config_file.replace("batt_not_fixed.json", "training_simul.json")
    with open(new_config_file, "w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    for i in [1, 2, 3, 5]:
        data_folder = f"data/houses/house_{i}/2023-09-08_10:00:00_2023-10-27_16:00:00"
        generate_config_train(data_folder)
