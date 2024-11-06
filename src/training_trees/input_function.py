import datetime
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from copy import deepcopy


def reduced_state_and_min_max_price(param_evaluation):
    inputs_list, input_info_list = reduced_state_input_norm_price(param_evaluation)
    inputs = inputs_list[0]
    input_info = input_info_list[0]

    microgrid = param_evaluation["microgrid"]
    time_step = microgrid.time_step
    utc_time = microgrid.utc_datetime

    if "switch_hour" in param_evaluation.keys():
        switch_hour = param_evaluation["switch_hour"]
        start_dt = utc_time.replace(hour=switch_hour, minute=0, second=0)

        if utc_time.hour < switch_hour:
            start_dt -= datetime.timedelta(days=1)

        end_dt = start_dt + datetime.timedelta(days=1)

        day_prices_24h = (
            microgrid.environments[0]
            .env_values["day_ahead_price"]
            .get_forecast(start_dt, end_dt)
        )["values"]

        index_now_steps = (utc_time - start_dt) // time_step

        num_steps_per_h = int(3600 / time_step.total_seconds())
        index_now_h = index_now_steps // num_steps_per_h

        day_prices_24h_in_hours = [
            p for i, p in enumerate(day_prices_24h) if i % num_steps_per_h == 0
        ]
        loc_min = list(argrelextrema(np.array(day_prices_24h_in_hours), np.less)[0])
        loc_max = list(argrelextrema(np.array(day_prices_24h_in_hours), np.greater)[0])

        if day_prices_24h_in_hours[0] < day_prices_24h_in_hours[1]:
            loc_min = [0] + loc_min
        else:
            loc_max = [0] + loc_max
        if day_prices_24h_in_hours[-2] > day_prices_24h_in_hours[-1]:
            loc_min = loc_min + [len(day_prices_24h_in_hours) - 1]
        else:
            loc_max = loc_max + [len(day_prices_24h_in_hours) - 1]

        if index_now_h in loc_min:
            next_loc_min = 0
        else:
            for loc in loc_min:
                if loc > index_now_h:
                    next_loc_min = loc - index_now_steps / num_steps_per_h
                    break
                else:
                    next_loc_min = 24
        if index_now_h in loc_max:
            next_loc_max = 0
        else:
            for loc in loc_max:
                if loc > index_now_h:
                    next_loc_max = loc - index_now_steps / num_steps_per_h
                    break
                else:
                    next_loc_max = 24

        inputs.append(next_loc_min)
        input_info.append(["next_loc_min (h)", [0, 24]])

        inputs.append(next_loc_max)
        input_info.append(["next_loc_max (h)", [0, 24]])

        global_min_index = np.argmin(day_prices_24h_in_hours)
        global_max_index = np.argmax(day_prices_24h_in_hours)

        if global_min_index == index_now_h:
            global_min_relative = 0
        else:
            global_min_relative = global_min_index - index_now_steps / num_steps_per_h
        if global_max_index == index_now_h:
            global_max_relative = 0
        else:
            global_max_relative = global_max_index - index_now_steps / num_steps_per_h

        inputs.append(global_min_relative)
        input_info.append(["global_min_relative (h)", [-24, 24]])

        inputs.append(global_max_relative)
        input_info.append(["global_max_relative (h)", [-24, 24]])

        return [inputs, inputs], [input_info, input_info]


def reduced_state_input_norm_price(param_evaluation):
    # Time of day utc in minutes
    # Day of week
    # Soc of the battery
    # PV production
    # Consumption
    # Price now
    # Soc of the ev

    inputs_list, input_info_list = reduced_state_input(param_evaluation)
    inputs = inputs_list[0]
    input_info = input_info_list[0]

    microgrid = param_evaluation["microgrid"]
    time_step = microgrid.time_step
    utc_time = microgrid.utc_datetime

    if "switch_hour" in param_evaluation.keys():
        switch_hour = param_evaluation["switch_hour"]
        start_dt = utc_time.replace(hour=switch_hour, minute=0, second=0)

        if utc_time.hour < switch_hour:
            start_dt -= datetime.timedelta(days=1)

        end_dt = start_dt + datetime.timedelta(days=1)

        day_prices_24h = (
            microgrid.environments[0]
            .env_values["day_ahead_price"]
            .get_forecast(start_dt, end_dt)
        )["values"]

        price_now = (
            microgrid.environments[0]
            .env_values["day_ahead_price"]
            .get_forecast(utc_time, utc_time + time_step)["values"][0]
        )

        norm_price = price_now - min(day_prices_24h)

        inputs.append(norm_price)
        input_info.append(["norm_price_now", [0.0, 0.5]])

    return [inputs, inputs], [input_info, input_info]


def reduced_state_no_real_price(param_evaluation):
    inputs_list, input_info_list = reduced_state_input_norm_price(param_evaluation)

    for i, indiv_info_list in enumerate(input_info_list):
        for j, info in enumerate(indiv_info_list):
            if "price_now" == info[0]:
                del inputs_list[i][j]
                del input_info_list[i][j]

    return inputs_list, input_info_list


def reduced_state_input(param_evaluation):
    # Time of day utc in minutes
    # Day of week
    # Soc of the battery
    # PV production
    # Consumption
    # Price now
    # Soc of the ev

    microgrid = param_evaluation["microgrid"]
    time_step = microgrid.time_step
    inputs = list()
    input_info = list()

    utc_time = microgrid.utc_datetime
    hour = utc_time.hour
    minute = utc_time.minute
    time_of_day = hour + minute / 60
    inputs.append(time_of_day)
    input_info.append(["time of day (h)", [0, 24]])

    day_of_week = utc_time.weekday()
    inputs.append(day_of_week)
    input_info.append(["day_of_week", [0, 7]])

    for asset in microgrid.assets:
        if asset.name.startswith("Battery"):
            soc = asset.soc
            battery = asset
        elif asset.name.startswith("SolarPv"):
            solar_pv = asset
        elif asset.name.startswith("Consumer"):
            consumer = asset
        elif asset.name.startswith("Charger"):
            charger = asset

    inputs.append(soc)
    input_info.append(["soc", [0, 1]])

    assets = [battery, solar_pv, consumer, charger]

    for asset in assets:
        if len(microgrid.power_hist) == 0:
            asset_power = 0
        elif asset.name in microgrid.power_hist[0]:
            asset_power = microgrid.power_hist[0][asset.name][-1].electrical
        else:
            asset_power = microgrid.power_hist[1][asset.name][-1].electrical

        max_prod = asset.max_production_power
        max_cons = asset.max_consumption_power
        inputs.append(asset_power)
        input_info.append([f"{asset.name}_power", [-max_cons, max_prod]])

    price_now = (
        microgrid.environments[0]
        .env_values["day_ahead_price"]
        .get_forecast(utc_time, utc_time + time_step)["values"][0]
    )

    inputs.append(price_now)
    input_info.append(["price_now", [-0.1, 0.5]])

    bug_ev_start = datetime.datetime(2024, 4, 11, 15, 0, 0)
    bug_ev_end = datetime.datetime(2024, 4, 15, 15, 0, 0)
    naive_time = utc_time.replace(tzinfo=None)

    if bug_ev_start <= naive_time < bug_ev_end:
        step_soc = charger.soc / 100
    else:
        step_soc = charger.soc

    inputs.append(step_soc)
    input_info.append(["soc_ev", [0, 1]])

    return [inputs, inputs], [input_info, input_info]


def complete_state_input(param_evaluation):
    # Time of day utc in minutes
    # Day of week
    # Soc of the battery
    # PV production
    # Consumption
    # Price now
    # EV energy to charge
    # EV detention time remaining
    # Price for the 24 hours

    inputs_list, input_info_list = reduced_state_input(param_evaluation)
    inputs = inputs_list[0]
    input_info = input_info_list[0]

    microgrid = param_evaluation["microgrid"]
    switch_hour = param_evaluation["switch_hour"]
    time_step = microgrid.time_step
    utc_time = microgrid.utc_datetime

    start_dt = utc_time.replace(hour=switch_hour, minute=0, second=0)

    if utc_time.hour < switch_hour:
        start_dt -= datetime.timedelta(days=1)

    end_dt = start_dt + datetime.timedelta(days=1)
    day_prices_24h = (
        microgrid.environments[0]
        .env_values["day_ahead_price"]
        .get_forecast(start_dt, end_dt)
    )["values"]

    num_steps_per_h = int(3600 / time_step.total_seconds())
    day_prices_24h_in_hours = [
        p for i, p in enumerate(day_prices_24h) if i % num_steps_per_h == 0
    ]

    # Price with min set to 0
    for i in range(24):
        hour = (switch_hour + i) % 24
        # norm_price = day_prices_24h_in_hours[i] - min(day_prices_24h_in_hours)
        price = day_prices_24h_in_hours[i]
        inputs.append(price)
        input_info.append([f"price_{hour}h", [-0.1, 0.5]])

    return [inputs, inputs], [input_info, input_info]


def house_input_func(param_evaluation):
    # Hour UTC
    # Minute
    # Day of week
    # Price next hour
    # Average price next 3 hours
    # Price difference
    # SOC of battery
    # PV production
    # Consumption
    # Time to next local maximum
    # Time to next local minimum
    # Difference with next local maximum
    # Difference with next local minimum
    # Time to golbal maximum
    # Time to golbal minimum
    # Difference with next golbal maximum
    # Difference with next golbal minimum
    microgrid = param_evaluation["microgrid"]
    inputs = list()
    input_info = list()

    utc_time = microgrid.utc_datetime
    hour = utc_time.hour
    inputs.append(hour)
    input_info.append(["hour", [0, 24]])

    minute = utc_time.minute
    inputs.append(minute)
    input_info.append(["minute", [0, 60]])

    day_of_week = utc_time.weekday()
    inputs.append(day_of_week)
    input_info.append(["day_of_week", [0, 7]])

    start_dt = utc_time.replace(hour=0, minute=0, second=0)
    end_dt = start_dt + datetime.timedelta(days=1)
    cur_dt_index = (utc_time - start_dt) // microgrid.time_step

    day_prices_24h = (
        microgrid.environments[0]
        .env_values["day_ahead_price"]
        .get_forecast(start_dt, end_dt)
    )["values"]

    for asset in microgrid.assets:
        if asset.name.startswith("Battery"):
            soc = asset.soc
        elif asset.name.startswith("SolarPv"):
            solar_pv = asset
        elif asset.name.startswith("Consumer"):
            consumer = asset

    inputs.append(soc)
    input_info.append(["soc", [0, 1]])

    try:
        pv_power = microgrid.power_hist[0]["SolarPv_0"][-1].electric
    except:
        pv_power = 0
    prod_max = solar_pv.max_production_power
    inputs.append(pv_power)
    input_info.append(["pv_power", [0, prod_max]])

    try:
        cons_power = microgrid.power_hist[1]["Consumer_0"][-1].electric
    except:
        cons_power = 0
    cons_max = consumer.max_consumption_power
    inputs.append(cons_power)
    input_info.append(["cons_power", [-cons_max, 0]])

    price_min_ind = np.argmin(day_prices_24h)
    price_max_ind = np.argmax(day_prices_24h)
    to_global_min = price_min_ind - cur_dt_index
    to_global_max = price_max_ind - cur_dt_index

    num_steps_per_h = int(3600 / microgrid.time_step.total_seconds())
    day_prices_24h_in_hours = [
        p for i, p in enumerate(day_prices_24h) if i % num_steps_per_h == 0
    ]
    pos_signal = np.array(day_prices_24h_in_hours)
    neg_signal = -pos_signal

    local_mins = list(find_peaks(neg_signal)[0])
    if pos_signal[0] < pos_signal[1]:
        local_mins = [0] + local_mins
    if pos_signal[-2] > pos_signal[-1]:
        local_mins = local_mins + [len(pos_signal) - 1]

    for local_min in local_mins:
        index_of_min = local_min * 4
        if index_of_min >= cur_dt_index:
            break
    to_local_min_ind = index_of_min - cur_dt_index

    local_maxs = list(find_peaks(pos_signal)[0])

    if pos_signal[0] > pos_signal[1]:
        local_maxs = [0] + local_maxs
    if pos_signal[-2] < pos_signal[-1]:
        local_maxs = local_maxs + [len(pos_signal) - 1]

    for local_max in local_maxs:
        index_of_max = local_max * 4
        if index_of_max >= cur_dt_index:
            break
    to_local_max_ind = index_of_max - cur_dt_index

    inputs.append(to_local_min_ind / 4)
    input_info.append(["time to local min (h)", [-24, 24]])

    inputs.append(to_local_max_ind / 4)
    input_info.append(["time to local max (h)", [-24, 24]])

    inputs.append(to_global_min / 4)
    input_info.append(["time to global max (h)", [-24, 24]])

    inputs.append(to_global_max / 4)
    input_info.append(["time to global min (h)", [-24, 24]])

    return [inputs], [input_info]


def house_input_func_old(param_evaluation):
    # Hour (UTC or local ?)
    # Minute
    # Day of week
    # Price next hour
    # Average price next 3 hours
    # Price difference
    # SOC of battery
    # PV production
    # Consumption
    # Time to next local maximum
    # Time to next local minimum
    # Difference with next local maximum
    # Difference with next local minimum
    # Time to next golbal maximum
    # Time to next golbal minimum
    # Difference with next golbal maximum
    # Difference with next golbal minimum
    microgrid = param_evaluation["microgrid"]
    inputs = list()
    input_info = list()

    utc_time = microgrid.utc_datetime
    hour = utc_time.hour
    inputs.append(hour)
    input_info.append(["hour", [0, 24]])

    minute = utc_time.minute
    inputs.append(minute)
    input_info.append(["minute", [0, 60]])

    day_of_week = utc_time.weekday()
    inputs.append(day_of_week)
    input_info.append(["day_of_week", [0, 7]])

    start_dt = utc_time
    end_dt = start_dt + datetime.timedelta(days=1)
    day_prices_24h = (
        microgrid.environments[0]
        .env_values["day_ahead_price"]
        .get_forecast(start_dt, end_dt)
    )["values"]
    NUM_TIMESTEPS = 24 * 4
    if len(day_prices_24h) != NUM_TIMESTEPS:
        num_missing = NUM_TIMESTEPS - len(day_prices_24h)
        day_prices_24h = day_prices_24h + [np.mean(day_prices_24h)] * num_missing
    diff_cur_next = day_prices_24h[1] - day_prices_24h[0]
    inputs.append(diff_cur_next)
    input_info.append(["diff_cur_next", [-0.25, 0.25]])

    diff_cur = list()
    for i in [1, 3, 6]:
        diff_cur = day_prices_24h[i * 4] - day_prices_24h[0]
        inputs.append(diff_cur)
        input_info.append([f"diff_cur_{i}h", [-0.25, 0.25]])

    for asset in microgrid.assets:
        if asset.name.startswith("Battery"):
            soc = asset.soc
        elif asset.name.startswith("SolarPv"):
            solar_pv = asset
        elif asset.name.startswith("Consumer"):
            consumer = asset

    inputs.append(soc)
    input_info.append(["soc", [0, 1]])

    try:
        pv_power = microgrid.power_hist[0]["SolarPv_0"][-1].electric
    except:
        pv_power = 0
    prod_max = solar_pv.max_production_power
    inputs.append(pv_power)
    input_info.append(["pv_power", [0, prod_max]])

    try:
        cons_power = microgrid.power_hist[1]["Consumer_0"][-1].electric
    except:
        cons_power = 0
    cons_max = consumer.max_consumption_power
    inputs.append(cons_power)
    input_info.append(["cons_power", [-cons_max, 0]])

    start_dt_minus_1 = start_dt - datetime.timedelta(hours=1)
    day_prices_25h = (
        microgrid.environments[0]
        .env_values["day_ahead_price"]
        .get_forecast(start_dt_minus_1, end_dt)
    )["values"]

    local_min_ind, local_min_val = find_opt(day_prices_25h, local=True, maximum=False)
    local_max_ind, local_max_val = find_opt(day_prices_25h, local=True, maximum=True)
    global_min_ind, global_min_val = find_opt(
        day_prices_25h, local=False, maximum=False
    )
    global_max_ind, global_max_val = find_opt(day_prices_25h, local=False, maximum=True)

    diff_local_min = local_min_val - day_prices_24h[0]
    diff_local_max = local_max_val - day_prices_24h[0]
    diff_global_min = global_min_val - day_prices_24h[0]
    diff_global_max = global_max_val - day_prices_24h[0]

    inputs.append(local_min_ind / 4 - 1)
    input_info.append(["time to local min (h)", [-1, 24]])

    inputs.append(local_max_ind / 4 - 1)
    input_info.append(["time to local max (h)", [-1, 24]])

    inputs.append(global_min_ind / 4 - 1)
    input_info.append(["time to global max (h)", [-1, 24]])

    inputs.append(global_max_ind / 4 - 1)
    input_info.append(["time to global min (h)", [-1, 24]])

    inputs.append(diff_local_min)
    input_info.append(["price diff local min (h)", [-0.25, 0.25]])

    inputs.append(diff_local_max)
    input_info.append(["price diff local max (h)", [-0.25, 0.25]])

    inputs.append(diff_global_min)
    input_info.append(["price diff global max (h)", [-0.25, 0.25]])

    inputs.append(diff_global_max)
    input_info.append(["price diff global min (h)", [-0.25, 0.25]])

    return [inputs], [input_info]


def find_opt(signal, local=False, maximum=True):
    prev_val = signal[0]
    optimums_val = list()
    optimums_ind = list()
    prev_going_up = None
    for i, val in enumerate(signal):
        if val == prev_val:
            continue

        going_up = val > prev_val
        if prev_going_up is None:
            prev_going_up = going_up
            continue
        local_maximum = False
        local_minimum = False
        if prev_going_up and not going_up:
            local_maximum = True
        elif not prev_going_up and going_up:
            local_minimum = True

        if (local_maximum and maximum) or (local_minimum and not maximum):
            optimums_val.append(prev_val)
            optimums_ind.append(prev_ind)

        prev_going_up = going_up
        prev_val = val
        prev_ind = i

    if len(optimums_ind) == 0:
        firt_last = [signal[0], signal[-1]]
        first_last_ind = [0, len(signal) - 1]
        if maximum:
            opt_ind = np.argmax(firt_last)
        else:
            opt_ind = np.argmin(firt_last)
        return first_last_ind[opt_ind], firt_last[opt_ind]

    if local:
        return optimums_ind[0], optimums_val[0]
    else:
        if maximum:
            opt_ind = np.argmax(optimums_val)
        else:
            opt_ind = np.argmin(optimums_val)

        return optimums_ind[opt_ind], optimums_val[opt_ind]


if __name__ == "__main__":
    signal = [0, 1, 2, 4, 5, 3, 2, 3, -1, 3, 7, 1]
    print(find_opt(signal, local=True, maximum=True))
