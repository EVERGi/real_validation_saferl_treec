import pandas as pd
from sklearn import neighbors
import lightgbm as lgb
import optuna
import pickle
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import datetime
import os
import warnings


def to_datetime(dt_str):
    return datetime.datetime.strptime(
        dt_str.replace("Z", "+00:00"), "%Y-%m-%dT%H:%M:%S%z"
    ).replace(tzinfo=None)


def get_inputs_for_ev(start_dt, end_dt):
    ev_schedule_file = "data/houses/common_env/ev_schedule_2023_2024_switch_1500.csv"
    ev_schedule_df = pd.read_csv(ev_schedule_file)
    ev_schedule = ev_schedule_df.to_dict(orient="list")
    ev_schedule["datetime"] = [to_datetime(d) for d in ev_schedule["datetime"]]

    ev_schedule_data = {
        "datetime": [],
        "soc_init": [],
        "detention": [],
        "soc_final": [],
    }
    for i, dt in enumerate(ev_schedule["datetime"]):
        if start_dt <= dt < end_dt and ev_schedule["det_0 (/)"][i] != 0:
            ev_schedule_data["datetime"].append(dt)
            ev_schedule_data["soc_init"].append(ev_schedule["soc_i_0 (/)"][i])
            ev_schedule_data["detention"].append(ev_schedule["det_0 (/)"][i])
            ev_schedule_data["soc_final"].append(ev_schedule["soc_f_0 (/)"][i])

    return ev_schedule_data


def get_inputs_for_load(start_dt, end_dt):
    load_file = "data/houses/common_env/SFH19_2023_2024_15min_original.csv"
    load_df = pd.read_csv(load_file)
    load = load_df.to_dict(orient="list")
    load["datetime"] = [to_datetime(d) for d in load["datetime"]]

    load_data = {
        "datetime": [],
        "load": [],
    }
    for i, dt in enumerate(load["datetime"]):
        if start_dt <= dt < end_dt:
            load_data["datetime"].append(dt)
            load_data["load"].append(load["Consumer_0_electric (kW)"][i])
    return load_data


def preprocess_dict(data):
    """
    Convert a dictionary of lists to a pandas dataframe
    """
    df = pd.DataFrame(data)
    df = df.set_index("datetime")
    # hour of the day (0-23)
    df["hour"] = df.index.hour
    df = df.sort_index()
    return df


# funciton to save model as pickle
def save_model(model, model_file):
    """
    Save a model as a pickle file
    """
    with open(model_file, "wb") as f:
        pickle.dump(model, f)


def train_ev_model(ev_schedule_data):
    """
    Train a forecasting model to predict detention and soc_final from datetime and soc_init
    """
    ev_schedule_df = preprocess_dict(ev_schedule_data)
    # print(ev_schedule_data)
    # train a model for n nearest neighbors based on cosine similarity
    neigh = neighbors.KNeighborsRegressor(n_neighbors=3, metric="cosine")
    detention_model = neigh.fit(
        ev_schedule_df[["soc_init", "hour"]].values,
        ev_schedule_df["detention"],
    )
    neigh2 = neighbors.KNeighborsRegressor(n_neighbors=3, metric="cosine")
    soc_final_model = neigh2.fit(
        ev_schedule_df[["soc_init", "hour"]].values, ev_schedule_df["soc_final"]
    )
    # you predict using the mean of the n nearest neighbors
    # detention_preds = detention_model.predict([[0.5, 19]])
    # soc_final_preds = soc_final_model.predict([[0.5, 19]])
    # print(detention_preds)
    # print(soc_final_preds)
    return detention_model, soc_final_model


def objective(
    trial,
    train_data,
    test_data,
    lgbm_params,
    target_name="SolarPv_0 (kW)",
    cat_feats=["hour"],
):
    params = {
        "max_depth": trial.suggest_categorical("max_depth", lgbm_params["max_depth"]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", lgbm_params["learning_rate"]
        ),
        "num_leaves": trial.suggest_categorical(
            "num_leaves", lgbm_params["num_leaves"]
        ),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", lgbm_params["colsample_bytree"]
        ),
        "min_child_samples": trial.suggest_categorical(
            "min_child_samples", lgbm_params["min_child_samples"]
        ),
        "verbosity": -1,
    }
    lgbm = lgb.LGBMRegressor(**params)
    lgbm.fit(
        train_data.drop(target_name, axis=1),
        train_data[target_name],
        categorical_feature=cat_feats,
    )
    preds = lgbm.predict(test_data.drop(target_name, axis=1))
    mae = np.mean(np.abs(preds - test_data[target_name]))
    return mae


def train_solar_model(pv_file):
    """
    Train a forecasting model to predict solar generation from datetime and lagging values
    The model should output the next 96 values (24 hours)
    """
    solar_data = pd.read_csv(pv_file, index_col="datetime", parse_dates=True)
    # add 24 lags of solar generation
    for i in range(1, 97):
        solar_data[f"solar-{i}"] = solar_data["SolarPv_0 (kW)"].shift(i)
    # add time of day
    solar_data["hour"] = solar_data.index.hour
    # lgbm hyperparameter ranges
    lgbm_params = {
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "num_leaves": [32, 64, 128, 256],
        "max_depth": [2, 4, 6, 8, 10],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [1, 2, 3, 4, 5],
        "verbosity": -1,
    }
    # divide into train and test sets
    train_size = int(len(solar_data) * 0.8)
    train_data = solar_data.iloc[:train_size]
    test_data = solar_data.iloc[train_size:]
    # use optuna to find the best hyperparameters
    # Create a study
    study = optuna.create_study(direction="minimize")
    # Optimize the study
    study.optimize(
        lambda trial: objective(trial, train_data, test_data, lgbm_params), n_trials=50
    )
    best_params = study.best_params
    # print(best_params)
    # Create forward lags for the target variable
    for i in range(96):
        solar_data[f"forward_solar-{i}"] = solar_data["SolarPv_0 (kW)"].shift(-i)
    # Drop rows with missing values
    solar_data = solar_data.dropna()
    # Prepare your data
    X = solar_data.drop(
        ["SolarPv_0 (kW)"] + [f"forward_solar-{i}" for i in range(96)], axis=1
    )
    y = solar_data[[f"forward_solar-{i}" for i in range(96)]]

    # Create an instance of MultiOutputRegressor with LGBMRegressor as the base estimator
    mor = MultiOutputRegressor(lgb.LGBMRegressor(**best_params))

    # Fit the model to your dataneighbors
    solar_model = mor.fit(X, y, categorical_feature=["hour"])
    return solar_model


def train_load_model(load_data):
    """
    Train a forecasting model to predict load from datetime and lagging values
    The model should output the next 96 values (24 hours)
    """
    load_df = preprocess_dict(load_data)
    # add 96 lags of load
    for i in range(1, 97):
        load_df[f"load-{i}"] = load_df["load"].shift(i)
    # add day of week
    load_df["day_of_week"] = load_df.index.dayofweek
    # divide into train and test sets
    train_size = int(len(load_df) * 0.8)
    train_data = load_df.iloc[:train_size]
    test_data = load_df.iloc[train_size:]
    lgbm_params = {
        "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
        "num_leaves": [32, 64, 128, 256],
        "max_depth": [2, 4, 6, 8, 10],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [1, 2, 3, 4, 5],
        "verbosity": -1,
    }
    # use optuna to find the best hyperparameters
    # Create a study
    study = optuna.create_study(direction="minimize")
    # Optimize the study
    study.optimize(
        lambda trial: objective(
            trial,
            train_data,
            test_data,
            lgbm_params,
            target_name="load",
            cat_feats=["hour", "day_of_week"],
        ),
        n_trials=50,
    )
    best_params = study.best_params
    # print(best_params)
    # Create forward lags for the target variable
    for i in range(96):
        load_df[f"forward_load-{i}"] = load_df["load"].shift(-i)
    # Drop rows with missing values\
    load_df = load_df.dropna()
    # Prepare your data
    X = load_df.drop(["load"] + [f"forward_load-{i}" for i in range(96)], axis=1)
    y = load_df[[f"forward_load-{i}" for i in range(96)]]
    # Create an instance of MultiOutputRegressor with LGBMRegressor as the base estimator
    mor = MultiOutputRegressor(lgb.LGBMRegressor(**best_params))
    # Fit the model to your data
    load_model = mor.fit(X, y, categorical_feature=["hour", "day_of_week"])
    return load_model


def train_all_models(start_dt, end_dt, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    start_str = start_dt.strftime("%Y-%m-%d_%H%M")
    end_str = end_dt.strftime("%Y-%m-%d_%H%M")

    ev_schedule_data = get_inputs_for_ev(start_dt, end_dt)
    ev_model_file_detention = f"{save_dir}ev_model_detention_{start_str}_{end_str}.pkl"
    ev_model_file_soc = f"{save_dir}ev_model_soc_{start_str}_{end_str}.pkl"

    print("Training EV models")
    ev_model_detention, ev_model_soc = train_ev_model(ev_schedule_data)
    save_model(ev_model_detention, ev_model_file_detention)
    save_model(ev_model_soc, ev_model_file_soc)

    load_data = get_inputs_for_load(start_dt, end_dt)
    load_model_file = f"{save_dir}load_model_{start_str}_{end_str}.pkl"
    print("Training electric load model")
    load_model = train_load_model(load_data)
    save_model(load_model, load_model_file)

    print("Training solar PV models")
    houses = [1, 2, 3, 5]
    for house in houses:
        pv_file = (
            f"data/houses/house_{house}/{start_str}_{end_str}/environment/solar.csv"
        )
        pv_model_file = f"{save_dir}pv_model_{start_str}_{end_str}_{house}.pkl"
        pv_model = train_solar_model(pv_file)
        save_model(pv_model, pv_model_file)


def paper_forecasting_train(save_dir="data/new_models/forecasting/"):
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    start_dt = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end_dt = datetime.datetime(2024, 4, 1, 0, 0, 0)
    train_all_models(start_dt, end_dt, save_dir)

    print(
        f"Training complete, all forecasting models generated and saved to {save_dir}"
    )


if __name__ == "__main__":
    paper_forecasting_train()
