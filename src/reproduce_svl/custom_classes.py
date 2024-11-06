import pytz
from simugrid.rewards.reward import Reward
from simugrid.assets import PublicGrid
from simugrid.management.rational import RationalManager
import datetime
from treec.utils import denormalise_input, normalise_input
import random


class DayAheadEngie(Reward):
    def __init__(self):
        list_KPI = ["day_ahead", "trans_dis", "capacity", "year_fixed", "opex"]
        Reward.__init__(self, list_KPI)

        # self.imbalance_thresh = 100 # kW
        # TODO Estimation, can be improved, price brussels engie June 2023

        self.prev_cap_tariff = 0
        self.cur_peak_power = 2.5

    def calculate_kpi(self):
        microgrid = self.microgrid

        time_step_h = microgrid.time_step.total_seconds() / 3600

        for asset in microgrid.assets:
            if isinstance(asset, PublicGrid):
                env = asset.parent_node.environment
                kwh_offtake_cost = env.env_values["kwh_offtake_cost"].value
                capacity_tariff = env.env_values["capacity_tariff"].value  # €/kW/month
                year_cost = env.env_values["year_cost"].value
                injection_extra = env.env_values["injection_extra"].value
                offtake_extra = env.env_values["offtake_extra"].value

                grid_energy = asset.power_output.electrical * time_step_h

                day_ahead_price = env.env_values["day_ahead_price"].value

                if grid_energy > 0:
                    day_ahead_price = day_ahead_price + offtake_extra
                    if day_ahead_price > 0:
                        day_ahead_cost = grid_energy * day_ahead_price * 1.06
                    else:
                        day_ahead_cost = grid_energy * day_ahead_price
                    self.KPIs["day_ahead"] -= day_ahead_cost
                    self.KPIs["trans_dis"] -= grid_energy * kwh_offtake_cost
                else:
                    day_ahead_price = day_ahead_price + injection_extra
                    day_ahead_cost = grid_energy * day_ahead_price
                    self.KPIs["day_ahead"] -= day_ahead_cost
                self.calc_capacity_tarriff(
                    asset.power_output.electrical, capacity_tariff
                )
                time_step_prop_year = time_step_h / (365 * 24)
                self.KPIs["year_fixed"] -= time_step_prop_year * year_cost

                self.KPIs["opex"] = (
                    self.KPIs["day_ahead"]
                    + self.KPIs["trans_dis"]
                    + self.KPIs["capacity"]
                    + self.KPIs["year_fixed"]
                )

    def calc_capacity_tarriff(self, grid_power, capacity_tariff):
        utc_datetime = self.microgrid.utc_datetime
        time_step = self.microgrid.time_step

        start_month = datetime.datetime(utc_datetime.year, utc_datetime.month, 1, 0, 0)
        start_month = start_month.replace(tzinfo=pytz.utc)
        # Get the first day of next month
        start_next_month = (start_month + datetime.timedelta(days=32)).replace(day=1)

        is_new_month = utc_datetime == start_month
        if is_new_month:
            self.prev_cap_tariff = self.KPIs["capacity"]
            self.cur_peak_power = 2.5

        if grid_power > self.cur_peak_power:
            self.cur_peak_power = grid_power
        # Number of seconds in current month
        tot_sec_month = (start_next_month - start_month).total_seconds()
        # Get the time since beginning of the month or simulation
        if start_month >= self.microgrid.start_time:
            start_prop = start_month
        else:
            start_prop = self.microgrid.start_time

        sec_since_startprop = ((utc_datetime + time_step) - start_prop).total_seconds()

        use_prop_month = False
        if use_prop_month:
            prop_month = sec_since_startprop / tot_sec_month
        else:
            prop_month = 1
        self.KPIs["capacity"] = (
            self.prev_cap_tariff - prop_month * self.cur_peak_power * capacity_tariff
        )


class DayAheadPrice(Reward):
    def __init__(self, calc_imbalance=True):
        list_KPI = ["day_ahead", "trans_dis", "opex"]
        if calc_imbalance:
            list_KPI += "imbalance"
        Reward.__init__(self, list_KPI)

        # self.imbalance_thresh = 100 # kW
        # TODO Estimation, can be improved, price brussels engie June 2023
        self.calc_imbalance = calc_imbalance

    def calculate_kpi(self):
        microgrid = self.microgrid

        time_step_h = microgrid.time_step.total_seconds() / 3600
        schedule = microgrid.management_system.day_ahead_schedule

        if self.microgrid.utc_datetime in schedule[0].keys():
            scheduled_energy = schedule[0][self.microgrid.utc_datetime]
        else:
            scheduled_energy = 100

        for asset in microgrid.assets:
            if isinstance(asset, PublicGrid):
                env = asset.parent_node.environment
                grid_energy = asset.power_output.electrical * time_step_h
                day_ahead_price = env.env_values["day_ahead_price"].value
                imbalance_price = env.env_values["imbalance_price"].value
                day_ahead_cost, imbalance_cost = self.calculate_day_ahead(
                    day_ahead_price, imbalance_price, grid_energy, scheduled_energy
                )
                self.KPIs["day_ahead"] -= day_ahead_cost
                if self.calc_imbalance:
                    self.KPIs["imbalance"] -= imbalance_cost
                if grid_energy > 0:
                    self.KPIs["trans_dis"] -= grid_energy * self.trans_dis_cost
                self.KPIs["opex"] = self.KPIs["day_ahead"] + self.KPIs["trans_dis"]
                if self.calc_imbalance:
                    self.KPIs["opex"] += self.KPIs["imbalance"]

    def calculate_day_ahead(
        self, day_ahead_price, imbalance_price, grid_energy, scheduled_energy
    ):
        if self.calc_imbalance:
            scheduled_cost = scheduled_energy * day_ahead_price

            imbalance_cost = imbalance_price * (grid_energy - scheduled_energy)
        else:
            scheduled_cost = grid_energy * day_ahead_price
            imbalance_cost = 0
        return scheduled_cost, imbalance_cost


class DayAheadManager(RationalManager):
    def __init__(self, microgrid, schedule_method="naive", timezone="Europe/Brussels"):
        super().__init__(microgrid)
        self.day_ahead_schedule = [dict() for _ in self.consumers]
        self.local_tz = pytz.timezone(timezone)
        self.schedule_method = schedule_method

    def simulate_step(self):
        local_time = self.microgrid.utc_datetime.astimezone(self.local_tz)
        if local_time.hour == 14 and local_time.minute == 0 and local_time.second == 0:
            self.set_next_schedule()

        super().simulate_step()

    def set_next_schedule(self):
        time_step_h = self.microgrid.time_step.total_seconds() / 3600

        cur_datetime = self.microgrid.utc_datetime.astimezone(self.local_tz).replace(
            tzinfo=None
        )
        next_day = cur_datetime + datetime.timedelta(days=1)
        after_next_day = next_day + datetime.timedelta(days=1)

        next_midnight = self.local_tz.localize(
            next_day.replace(hour=0, minute=0, second=0)
        )
        after_next_midnight = self.local_tz.localize(
            after_next_day.replace(hour=0, minute=0, second=0)
        )

        utc_next_midnight = next_midnight.astimezone(pytz.utc)
        utc_after_next_midnight = after_next_midnight.astimezone(pytz.utc)

        for i, consumer in enumerate(self.consumers):
            if self.schedule_method == "perfect":
                forec_cons = consumer.get_forecast(
                    utc_next_midnight, utc_after_next_midnight, "perfect"
                )
            elif self.schedule_method == "naive":
                forec_cons = consumer.get_forecast(
                    utc_next_midnight, utc_after_next_midnight, "naive"
                )

            for j, utc_date in enumerate(forec_cons["electric_power"]["datetime"]):
                self.day_ahead_schedule[i][utc_date] = (
                    forec_cons["electric_power"]["values"][j] * time_step_h
                )


class TreeManager(RationalManager):
    def __init__(self, microgrid, trees, input_func):
        super().__init__(microgrid)
        self.trees = trees

        self.input_func = input_func

    def get_actions(self):
        actions = list()

        leaf_indexes = list()

        input_dict = {"microgrid": self.microgrid}
        features, input_info = self.input_func(input_dict)
        for i, tree in enumerate(self.trees):
            node = tree.get_action_with_unnormalised_input(features[i])

            leaf_indexes.append(tree.node_stack.index(node))

            actions.append(node.value)

        return actions, leaf_indexes


def bound(low, high, value):
    return max(low, min(high, value))


class SVLBattCharg(RationalManager):
    def __init__(self, microgrid):
        super().__init__(microgrid)
        self.battery = self.batteries[0]
        self.charger = self.chargers[0]
        self.batteries = []
        self.chargers = []
        self.grid = self.public_grid[0]
        self.pv = self.renewable_assets[0]
        self.consumer = self.consumers[0]
        self.hour_day_switch = None

    def set_hour_day_switch(self, hour_day_switch):
        self.hour_day_switch = hour_day_switch

    def execute_switch_time(self):
        microgrid = self.microgrid
        time_step = microgrid.time_step
        time_step_h = time_step.total_seconds() / 3600
        current_time = microgrid.utc_datetime
        time_switch = current_time.replace(hour=self.hour_day_switch, minute=0)

        if time_switch <= current_time:
            time_switch += datetime.timedelta(days=1)
        MAX_BUFFER_SEC = 60 * 60
        MIN_BUFFER_SEC = 3 * 60
        soc = self.battery.soc
        buffer_seconds = (1 - soc) * (MAX_BUFFER_SEC - MIN_BUFFER_SEC) + MIN_BUFFER_SEC
        buffer_td = datetime.timedelta(seconds=buffer_seconds)

        time_to_switch = time_switch - current_time

        batt_power = self.battery.power_output.electrical
        next_soc = self.battery.soc + self.battery.soc_change(batt_power, time_step)
        time_step_charge_max = time_to_switch - time_step
        time_step_charge_max = time_step_charge_max - buffer_td

        FULL_CHARGE = 1.0

        max_charge_power = -self.battery.max_consumption_power
        soc_change_max_charge = self.battery.soc_change(
            max_charge_power, time_step_charge_max
        )

        min_next_soc = FULL_CHARGE - soc_change_max_charge

        if next_soc < min_next_soc:
            diff_to_min = min_next_soc - self.battery.soc
            if diff_to_min > 0:
                enforced_power = (
                    -diff_to_min
                    / time_step_h
                    * self.battery.size
                    / self.battery.charge_eff
                )
            else:
                enforced_power = (
                    -diff_to_min
                    / time_step_h
                    * self.battery.size
                    * self.battery.disch_eff
                )
            bounded_power = max(self.battery.power_limit_low.electrical, enforced_power)
            batt_power = self.battery.power_output.electrical

            diff_power = batt_power - bounded_power
            pro = self.grid
            cons = self.battery

            self.exec_power_trans(pro, cons, power_send=diff_power)

    def simulate_step(self):
        actions, leaf_indexes = self.get_actions()
        power_battery = actions[0]
        power_charger = actions[1]

        bounded_batt = bound(
            self.battery.power_limit_low.electrical,
            self.battery.power_limit_high.electrical,
            power_battery,
        )

        MAX_BUFFER_SEC = 4 * 60 * 60
        MIN_BUFFER_SEC = 3 * 60
        soc_left = self.charger.soc_f - self.charger.soc
        buffer_seconds = soc_left * (MAX_BUFFER_SEC - MIN_BUFFER_SEC) + MIN_BUFFER_SEC
        buffer_h = buffer_seconds / 3600

        charger_low, charger_high = self.charger.get_powers_to_reach_soc_final(
            buffer_time_h=buffer_h
        )
        if power_charger > 1.0:
            pass

        bounded_charger = bound(charger_low, charger_high, power_charger)

        if bounded_batt > 0:
            pro = self.battery
            cons = self.grid
        else:
            pro = self.grid
            cons = self.battery
        self.exec_power_trans(pro, cons, power_send=abs(bounded_batt))

        self.exec_power_trans(self.grid, self.charger, power_send=abs(bounded_charger))

        if self.hour_day_switch is not None:
            self.execute_switch_time()

        self.execute_safety_layer()

        super().simulate_step()

        try:
            batt_power = self.microgrid.power_hist[1]["Battery_0"][-1].electrical
        except IndexError:
            batt_power = self.microgrid.power_hist[0]["Battery_0"][-1].electrical
        charger_power = self.microgrid.power_hist[0]["Charger_0"][-1].electrical
        return leaf_indexes

    def execute_safety_layer(self):
        cur_time = self.microgrid.utc_datetime.replace(tzinfo=None)
        start_previous_max = datetime.datetime(2024, 1, 1, 0)
        end_previous_max = datetime.datetime(2024, 5, 21, 15, 00)
        if start_previous_max <= cur_time < end_previous_max:
            MAX_GRID_POWER = 9.2  # kW
        else:
            MAX_GRID_POWER = 8.7
        grid_power = -(
            self.pv.power_limit_high.electrical
            + self.consumer.power_limit_low.electrical
            + self.charger.power_output.electrical
            + self.battery.power_output.electrical
        )

        excess_power = grid_power - MAX_GRID_POWER

        assets_to_reduce = [self.battery, self.charger]

        power_assets = [asset.power_output.electrical for asset in assets_to_reduce]
        power_limits = [asset.power_limit_high.electrical for asset in assets_to_reduce]

        avail_power = [
            power_limit - power
            for power, power_limit in zip(power_assets, power_limits)
        ]

        asset_power_trans = [0 for _ in assets_to_reduce]
        if excess_power > 0:
            if min(avail_power) > excess_power / len(assets_to_reduce):
                for i, asset in enumerate(assets_to_reduce):
                    asset_power_trans[i] = excess_power / len(assets_to_reduce)
            else:
                min_asset_index = avail_power.index(min(avail_power))
                asset_power_trans[min_asset_index] = avail_power[min_asset_index]
                power_remaining = excess_power - min(avail_power)

                max_asset_index = int(not min_asset_index)
                asset_power_trans[max_asset_index] = min(
                    power_remaining, avail_power[max_asset_index]
                )
            for i, asset in enumerate(assets_to_reduce):
                power_send = asset_power_trans[i]
                self.exec_power_trans(asset, self.grid, power_send=power_send)


class NoEMSSVL(SVLBattCharg):
    def __init__(self, microgrid):
        super().__init__(microgrid)

    def get_actions(self):
        batt_power = self.battery.power_limit_high.electrical
        return [batt_power, 0], []


class RandomEMS(SVLBattCharg):
    def __init__(self, microgrid):
        super().__init__(microgrid)

    def get_actions(self):
        battery = self.battery
        max_batt = self.battery.max_production_power
        min_batt = -self.battery.max_consumption_power

        charger = self.charger
        max_charger = charger.max_production_power
        min_charger = -charger.max_consumption_power

        batt_power = denormalise_input(random.random(), min_batt, max_batt)
        charge_power = denormalise_input(random.random(), min_charger, max_charger)

        return [batt_power, charge_power], []


class RLApproximation(SVLBattCharg):
    def __init__(self, microgrid):
        super().__init__(microgrid)

    def get_actions(self):
        cur_time = self.microgrid.utc_datetime.replace(tzinfo=None)

        period_one_end = datetime.datetime(2024, 4, 14, 15)
        period_two_start = datetime.datetime(2024, 4, 16, 15)
        period_two_end = datetime.datetime(2024, 4, 17, 15)
        in_period_one = cur_time < period_one_end
        in_period_two = period_two_start <= cur_time < period_two_end

        if in_period_one or in_period_two:
            action = self.random_action()
        else:
            action = self.discharge_action()
        return action

    def random_action(self):
        battery = self.battery
        max_batt = battery.max_production_power
        min_batt = -battery.max_consumption_power

        charger = self.charger
        max_charger = charger.max_production_power
        min_charger = -charger.max_consumption_power

        batt_power = denormalise_input(random.random(), min_batt, max_batt)
        charge_power = denormalise_input(random.random(), min_charger, max_charger)

        return [batt_power, charge_power], []

    def discharge_action(self):
        battery = self.battery
        max_batt = battery.max_production_power

        charger = self.charger
        max_charger = charger.max_production_power
        min_charger = -6

        charge_power = denormalise_input(random.random(), min_charger, max_charger)

        return [max_batt, charge_power], []


class RationalSVL(SVLBattCharg):
    def __init__(self, microgrid):
        super().__init__(microgrid)

    def get_actions(self):
        for asset in self.microgrid.assets:
            if asset.name == "SolarPv_0":
                solar_pv = asset
            elif asset.name == "Consumer_0":
                consumer = asset
        charge_power = self.charger.power_limit_low.electrical
        pv_power = solar_pv.power_limit_high.electrical
        cons_power = consumer.power_limit_low.electrical
        batt_power = -(charge_power + pv_power + cons_power)

        actions = [batt_power, charge_power]
        return actions, []


class SVLTreeBattCharg(SVLBattCharg):
    def __init__(self, microgrid, trees, input_func, params_evaluation):
        super().__init__(microgrid)
        self.trees = trees
        self.input_func = input_func

    def get_actions(self):
        actions = list()

        leaf_indexes = list()

        input_dict = {"microgrid": self.microgrid}
        if self.hour_day_switch is not None:
            input_dict["switch_hour"] = self.hour_day_switch
        features, input_info = self.input_func(input_dict)
        for i, tree in enumerate(self.trees):
            node = tree.get_action_with_unnormalised_input(features[i])

            leaf_indexes.append(tree.node_stack.index(node))

            actions.append(node.value)
        battery = self.battery
        min_power_batt = -battery.max_consumption_power
        max_power_batt = battery.max_production_power

        actions[0] = denormalise_input(actions[0], min_power_batt, max_power_batt)

        charger = self.charger
        min_power_charger = -charger.max_consumption_power
        max_power_charger = charger.max_production_power

        actions[1] = denormalise_input(actions[1], min_power_charger, max_power_charger)

        return actions, leaf_indexes


class SVLTreeBattChargRule(SVLTreeBattCharg):
    def __init__(self, microgrid, trees, input_func, params_evaluation):
        super().__init__(microgrid, trees, input_func, params_evaluation)

    def get_actions(self):
        actions, leaf_indexes = super().get_actions()
        charger_action = actions[1]

        battery = self.battery
        min_power_batt = -battery.max_consumption_power
        max_power_batt = battery.max_production_power
        batt_action = normalise_input(actions[0], min_power_batt, max_power_batt)

        RULE_LIMIT = 0.1
        if batt_action < RULE_LIMIT:
            consumer = self.consumer
            consumer_power = consumer.power_limit_low.electrical
            pv = self.pv
            pv_power = pv.power_limit_high.electrical
            charger_power = bound(
                self.charger.power_limit_low.electrical,
                self.charger.power_limit_high.electrical,
                charger_action,
            )

            batt_power = -(consumer_power + pv_power + charger_power)
            actions[0] = batt_power
        else:
            batt_action = (batt_action - RULE_LIMIT) / (1 - RULE_LIMIT)
            actions[0] = denormalise_input(batt_action, min_power_batt, max_power_batt)

        return actions, leaf_indexes


class CustomControl(SVLBattCharg):
    def __init__(self, microgrid, trees, input_func, params_evaluation):
        super().__init__(microgrid)
        self.trees = trees

        self.input_func = input_func
        self.house_num = params_evaluation["house_num"]

    def get_actions(self):
        actions = list()

        leaf_indexes = list()

        input_dict = {"microgrid": self.microgrid}
        features, input_info = self.input_func(input_dict)

        battery = self.battery
        min_power_batt = -battery.max_consumption_power
        max_power_batt = battery.max_production_power

        pv_power = features[0][4]
        cons_power = features[0][5]
        charger_power = features[0][6]

        price_now = features[0][31]

        batt_power = -(pv_power + cons_power + charger_power)

        # if price_now < 0.001:
        #    batt_power = min_power_batt
        if price_now > 0.2:
            batt_power = max_power_batt

        charger = self.charger
        min_power_charger = -charger.max_consumption_power
        max_power_charger = charger.max_production_power

        if price_now < 0.05:
            charge_power = -charger.max_consumption_power
        elif price_now > 0.08:
            charge_power = 0
        else:
            charge_power = -charger.max_consumption_power / 4

        actions = [batt_power, charge_power]

        return actions, leaf_indexes
