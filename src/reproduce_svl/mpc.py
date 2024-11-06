import sys
import os
import datetime


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from reproduce_svl.custom_classes import SVLBattCharg
import numpy as np
import pyomo.environ as pyo
from simugrid.assets.charger import first_order_fit
from forecasting.model_forecaster import (
    ModelMicroForecaster,
    ThreeMonthForecaster,
    BugForecaster,
)
import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)


class SVLMPCManager(SVLBattCharg):
    def __init__(
        self,
        microgrid,
        params_evaluation,
        forecast_mode="perfect",
    ):
        super().__init__(microgrid)

        self.forecast_quality = forecast_mode
        self.opt = pyo.SolverFactory(
            "ipopt",
            options={
                # "OutputFlag": 0,
                # "LogToConsole": 1,
            },
        )

        self.house_num = params_evaluation["house_num"]

        self.save_power = list()
        self.current_index = 0
        # self.optimisation_interval = 24 * 4
        self.optimisation_interval = 1
        self.control_mode = "battery"

        # If forecaster not in attribute list
        if not hasattr(self, "forecaster"):
            self.forecaster = ModelMicroForecaster(forecast_mode, self.house_num)

        env_values = self.microgrid.environments[0].env_values
        self.kwh_offtake_cost = env_values["kwh_offtake_cost"].value
        self.capacity_tariff = env_values["capacity_tariff"].value
        self.injection_extra = env_values["injection_extra"].value
        self.offtake_extra = env_values["offtake_extra"].value

        self.current_month_peak = 2.5
        self.total_cap_cost = 0

        self.update_peak_experiment = False

    def set_hour_day_switch(self, hour):
        self.hour_day_switch = hour
        self.forecaster.set_hour_day_switch(hour)

    def get_actions(self):
        if self.current_index % self.optimisation_interval == 0:
            self.current_index = 0

            self.do_optimisation()
            retry_counter = 0
            while self.model.batt_pos[0].value is None:
                retry_counter += 1
                self.do_optimisation(retry_counter)
                print(f"Retry counter: {retry_counter}")
                if retry_counter > 10:
                    break

            if self.model.batt_pos[0].value is None:
                print(self.microgrid.utc_datetime)
                with open("debug_model.out", "w+") as output_file:
                    self.model.pprint(output_file)

            batt_pos = [self.model.batt_pos[t].value for t in self.model.time]
            batt_neg = [self.model.batt_neg[t].value for t in self.model.time]

            charg_power = [self.model.charg_power[t].value for t in self.model.time]
            # batt_pos = pyo.value(self.model.batt_pos)
            soc = [self.model.soc[t].value for t in self.model.time]
            grid_power = [self.model.grid_power[t].value for t in self.model.time]
            # print(self.current_month_peak)
            # if max(grid_power) > 5:
            #    print("Grid power too high")
            #    cur_datetime = self.microgrid.utc_datetime
            #    with open(f"debug_model_{cur_datetime}.out", "w+") as output_file:
            #        self.model.pprint(output_file)

            price_obj = self.model.price_obj.value

            trans_dis_cost = [
                self.model.trans_dis_cost[t].value for t in self.model.time
            ]

            if self.control_mode == "battery":
                self.save_power = [
                    [batt_neg[i] + pos, charg_power[i]]
                    for i, pos in enumerate(batt_pos)
                ]
            elif self.control_mode == "grid":
                self.save_power = grid_power

        actions = self.save_power[self.current_index]

        self.current_index += 1

        # end_of_day_batt_charge(self.microgrid)

        return actions, list()

    def do_optimisation(self, retry_counter=0):
        self.update_grid_peak()

        self.model = pyo.ConcreteModel()

        self.build_sets()

        self.build_vars()
        pv_load, consumer_load, day_ahead, charge_info = self.prepare_info_constrs(
            retry_counter
        )
        self.build_constr(pv_load, consumer_load, day_ahead, charge_info)

        self.build_obj()

        self.opt.solve(self.model)

    def prepare_info_constrs(self, retry_counter=0):
        solar_pv = self.renewable_assets[0]
        consumer = self.consumers[0]
        charger = self.charger

        pv_load = self.forecaster.forecast_pv_load(self.microgrid)
        consumer_load = self.forecaster.forecast_consum_load(self.microgrid)
        day_ahead = self.forecaster.forecast_day_ahead(self.microgrid)
        charge_forec = self.forecaster.forecast_charger(self.microgrid)
        det_vals = charge_forec["det"]
        cap_vals = charge_forec["cap"]
        soc_i_vals = charge_forec["soc_i"]
        soc_f_vals = charge_forec["soc_f"]
        p_max_vals = charge_forec["p_max"]
        if retry_counter != 0 and det_vals[0] != 0:
            det_vals[0] += retry_counter

        prev_det = det_vals[0]
        for i, det_val in enumerate(det_vals[1:]):
            time_ind = i + 1
            det_val = int(det_val * 4)

            if prev_det != 0 and det_val == 0:
                det_vals[time_ind] = prev_det - 1
                if det_vals[time_ind] != 0:
                    cap_vals[time_ind] = cap_vals[time_ind - 1]
                    p_max_vals[time_ind] = p_max_vals[time_ind - 1]
            elif det_val == 0:
                cap_vals[time_ind] = 0
                p_max_vals[time_ind] = 0
            else:
                det_vals[time_ind] = det_val

            prev_det = det_vals[time_ind]

        imposed_soc_i = dict()
        imposed_soc_f = dict()
        for i, soc_i_val in enumerate(soc_i_vals):
            soc_f_val = soc_f_vals[i]
            det_val = det_vals[i]
            if soc_i_val != 0 or soc_f_val != 0:
                imposed_soc_i[i] = soc_i_val
                imposed_f_ind = i + det_val - 1
                if imposed_f_ind < len(det_vals):
                    imposed_soc_f[imposed_f_ind] = soc_f_val
                else:
                    soc_to_charge = soc_f_val - soc_i_val
                    time_to_end = len(det_vals) - i
                    new_soc_f = soc_i_val + time_to_end * soc_to_charge / det_val
                    imposed_soc_f[len(det_vals) - 1] = new_soc_f

        MAX_BUFFER_SEC = 4 * 60 * 60
        MIN_BUFFER_SEC = 3 * 60
        soc_left = self.charger.soc_f - self.charger.soc
        buffer_seconds = soc_left * (MAX_BUFFER_SEC - MIN_BUFFER_SEC) + MIN_BUFFER_SEC
        buffer_h = buffer_seconds / 3600

        _, forced_charge_power = charger.get_powers_to_reach_soc_final(
            buffer_time_h=buffer_h
        )
        charge_info = {
            "det": det_vals,
            "cap": cap_vals,
            "imposed_soc_i": imposed_soc_i,
            "imposed_soc_f": imposed_soc_f,
            "p_max": p_max_vals,
            "forced_charge_power": forced_charge_power,
        }
        # Relax problem
        if retry_counter != 0:
            print("Should never get here")
            # charge_info["forced_low_power"] += 0.2 * retry_counter
            prop_charge = (10 - retry_counter) / 10
            charg_diff = soc_f_vals[0] - soc_i_vals[0]
            soc_f_vals[0] = soc_i_vals[0] + prop_charge * charg_diff

        return pv_load, consumer_load, day_ahead, charge_info

    def update_grid_peak(self):
        utc_datetime = self.microgrid.utc_datetime

        is_new_month = (
            utc_datetime.day == 1
            and utc_datetime.hour == 0
            and utc_datetime.minute == 0
        )
        if len(self.microgrid.power_hist) == 0:
            prev_grid_power = 0
        else:
            prev_grid_power = self.microgrid.power_hist[0]["PublicGrid_0"][
                -1
            ].electrical

        if is_new_month:
            self.total_cap_cost += self.current_month_peak * self.capacity_tariff
            self.current_month_peak = 2.5
        elif prev_grid_power > self.current_month_peak:
            self.current_month_peak = prev_grid_power

    def simulate_step(self):
        super().simulate_step()

        # prev_power = self.microgrid.power_hist[0]["Battery_0"][-1].electrical
        # print("New step:")
        # print(f"Actual power: {prev_power}")
        # print(f"Wanted power: {batt_power}")

    def build_sets(self):
        horizon = self.calculate_horizon()
        model = self.model
        time = np.array([i for i in range(horizon)])

        model.time = pyo.Set(initialize=time)

    def calculate_horizon(self):
        time_step = self.microgrid.time_step
        cur_time = self.microgrid.utc_datetime
        switch_time = cur_time.replace(hour=self.hour_day_switch, minute=0)
        if switch_time <= cur_time:
            switch_time += datetime.timedelta(days=1)
        horizon = int((switch_time - cur_time) / time_step)

        return horizon

    def build_vars(self):
        model = self.model
        battery = self.battery
        charger = self.charger

        max_charge = -battery.max_consumption_power
        max_discharge = battery.max_production_power

        model.batt_pos = pyo.Var(
            model.time, domain=pyo.PositiveReals, bounds=(0, max_discharge)
        )
        model.batt_neg = pyo.Var(
            model.time, domain=pyo.NegativeReals, bounds=(max_charge, 0)
        )

        max_pow_charger = -charger.max_consumption_power
        model.charg_power = pyo.Var(
            model.time, domain=pyo.NegativeReals, bounds=(max_pow_charger, 0)
        )
        model.soc_charg = pyo.Var(
            model.time,
            domain=pyo.PositiveReals,
            bounds=(0, 1.0),
        )

        model.extra_vat_offtake = pyo.Var(model.time, domain=pyo.PositiveReals)
        model.extra_cost_injection = pyo.Var(model.time, domain=pyo.PositiveReals)
        model.extra_cost_offtake = pyo.Var(model.time, domain=pyo.PositiveReals)
        model.trans_dis_cost = pyo.Var(model.time, domain=pyo.NegativeReals)

        model.pv_power = pyo.Var(model.time, domain=pyo.PositiveReals)

        cur_time = self.microgrid.utc_datetime.replace(tzinfo=None)
        start_previous_max = datetime.datetime(2024, 1, 1, 0)
        end_previous_max = datetime.datetime(2024, 5, 21, 15, 00)
        if start_previous_max <= cur_time < end_previous_max:
            MAX_GRID_POWER = 9.2  # kW
        else:
            MAX_GRID_POWER = 8.7
        model.grid_power = pyo.Var(model.time, bounds=(-MAX_GRID_POWER, MAX_GRID_POWER))

        batt_capacity = battery.size
        model.soc = pyo.Var(
            model.time,
            domain=pyo.PositiveReals,
            bounds=(0, batt_capacity),
        )
        model.grid_peak = pyo.Var(bounds=(self.current_month_peak, None))
        model.cap_tariff_cost = pyo.Var(domain=pyo.PositiveReals)

        model.price_obj = pyo.Var(domain=pyo.Reals)

    def set_grid_peak_constr(self):
        model = self.model
        time = model.time

        def rule_grid_peak(model, t):
            return model.grid_peak >= model.grid_power[t]

        model.calc_grid_peak = pyo.Constraint(
            time,
            rule=rule_grid_peak,
        )

    def set_grid_power_constr(self, consumer_load):
        # Set variable with the summed power of consumptiona nd battery
        model = self.model
        time = model.time

        def rule_grid_power(model, t):
            batt_power = model.batt_pos[t] + model.batt_neg[t]
            return model.grid_power[t] == -(
                batt_power + model.pv_power[t] + consumer_load[t] + model.charg_power[t]
            )

        model.calc_grid_power = pyo.Constraint(
            time,
            rule=rule_grid_power,
        )

    def set_pv_power(self, pv_load):
        model = self.model
        time = model.time

        def rule_low_lim_pv(model, t):
            return model.pv_power[t] == pv_load[t]

        model.calc_low_lim_pv = pyo.Constraint(
            time,
            rule=rule_low_lim_pv,
        )
        """
        def rule_up_lim_pv(model, t):
            return model.pv_power[t] >= pv_load[t] - model.batt_pos[t]

        model.calc_up_lim_pv = pyo.Constraint(
            time,
            rule=rule_up_lim_pv,
        )
        """

    """
    def set_branch_lim(self):
        model = self.model

        time = model.time
        max_branch_power = self.microgrid.branches[0].max_power_electrical

        def rule_up_lim_branch(model, t):
            return (
                model.pv_power[t] + model.batt_pos[t] + model.batt_neg[t]
                <= max_branch_power
            )

        model.calc_up_lim_branch = pyo.Constraint(
            time,
            rule=rule_up_lim_branch,
        )

        def rule_low_lim_branch(model, t):
            return (
                model.pv_power[t] + model.batt_pos[t] + model.batt_neg[t]
                >= -max_branch_power
            )

        model.calc_low_lim_branch = pyo.Constraint(
            time,
            rule=rule_low_lim_branch,
        )
    """

    def trans_dis_constr(self):
        # Set price cost to be positive
        # forall s,t sum_b{batt_power[s,b,t]}-price_ind_cost[s,t]/price_cost[t]<=sum_b{-baseload[s,b,t]}
        # or sum_b{batt_power[s,b,t]+baseload[s,b,t]}*price_cost[t]<=price_ind_cost[s,t]
        model = self.model

        time = model.time

        time_step = self.microgrid.time_step
        t_s_hours = time_step.total_seconds() / 3600

        def rule_trans_dis_cost(model, t):
            return (
                model.trans_dis_cost[t]
                <= -model.grid_power[t] * t_s_hours * self.kwh_offtake_cost
            )

        model.calc_trans_dis_cost = pyo.Constraint(time, rule=rule_trans_dis_cost)

    def sum_price_cost_constr(self, day_ahead):
        time_step = self.microgrid.time_step
        t_s_hours = time_step.total_seconds() / 3600
        # Sum all price costs to get the final price cost per scenario.
        # forall s tot_price[s] = sum_[t]{price_ind_cost[s,t]}
        model = self.model
        time = model.time

        def rule_vat_offtake_calc(model, t):
            if day_ahead[t] + self.offtake_extra >= 0:
                return (
                    model.extra_vat_offtake[t]
                    >= model.grid_power[t]
                    * (day_ahead[t] + self.offtake_extra)
                    * t_s_hours
                    * 0.06
                )
            else:
                return model.extra_vat_offtake[t] == 0

        model.vat_offtake_calc = pyo.Constraint(time, rule=rule_vat_offtake_calc)

        def rule_extra_cost_offtake_calc(model, t):
            return (
                model.extra_cost_offtake[t]
                >= model.grid_power[t] * t_s_hours * self.offtake_extra
            )

        model.extra_cost_offtake_calc = pyo.Constraint(
            time,
            rule=rule_extra_cost_offtake_calc,
        )

        def rule_extra_cost_injection_calc(model, t):
            return (
                model.extra_cost_injection[t]
                >= -model.grid_power[t] * t_s_hours * -self.injection_extra
            )

        model.extra_cost_injection_calc = pyo.Constraint(
            time,
            rule=rule_extra_cost_injection_calc,
        )

        def rule_sum_price_cost(model):
            tot_price_cost = sum(model.trans_dis_cost[t] for t in time)
            tot_extra_offtake_injection = -sum(
                model.extra_vat_offtake[t]
                + model.extra_cost_injection[t]
                + model.extra_cost_offtake[t]
                for t in time
            )

            day_ahead_cost = -sum(
                model.grid_power[t] * t_s_hours * day_ahead[t] for t in time
            )
            day_ahead_cost += tot_extra_offtake_injection
            cap_cost = -model.grid_peak * self.capacity_tariff
            return model.price_obj == tot_price_cost + day_ahead_cost + cap_cost

        model.sum_price_cost = pyo.Constraint(rule=rule_sum_price_cost)

    def soc_constr(self):
        model = self.model

        time = model.time

        battery = self.battery

        time_step = self.microgrid.time_step
        t_s_hours = time_step.total_seconds() / 3600

        soc_init = battery.soc * battery.size
        charge_eff = battery.charge_eff
        disch_eff = battery.disch_eff

        def rule_soc_fixed(model, t):
            if t == 0:
                soc_prev = soc_init
            else:
                soc_prev = model.soc[t - 1]

            return (
                model.soc[t]
                == soc_prev
                - model.batt_pos[t] / charge_eff * t_s_hours
                - model.batt_neg[t] * disch_eff * t_s_hours
            )

        model.set_soc_fixed = pyo.Constraint(
            time,
            rule=rule_soc_fixed,
        )

        if self.hour_day_switch is not None:
            switch_step = len(time) - 1

            def rule_soc_switch(model, t):
                if t == switch_step:
                    return model.soc[t] == battery.size
                else:
                    return pyo.Constraint.Skip

            model.set_soc_switch = pyo.Constraint(
                time,
                rule=rule_soc_switch,
            )

    def charger_constr(self, charge_info):
        charger = self.charger
        model = self.model
        time_step = self.microgrid.time_step
        t_s_hours = time_step.total_seconds() / 3600

        cap = charge_info["cap"]
        p_max = charge_info["p_max"]
        soc_imposed_i = charge_info["imposed_soc_i"]
        soc_imposed_f = charge_info["imposed_soc_f"]
        det = charge_info["det"]
        eff = charger.eff

        charge_is_enforced = charge_info["forced_charge_power"] <= -0.01

        def imposed_soc_rule(model, t):
            if t in soc_imposed_f.keys():
                return model.soc_charg[t] >= soc_imposed_f[t]
            elif det[t] == 0:
                return model.soc_charg[t] == 0
            else:
                return pyo.Constraint.Skip

        model.imposed_soc = pyo.Constraint(model.time, rule=imposed_soc_rule)

        def enforced_charge_power_rule(model):
            if charge_is_enforced:
                return model.charg_power[0] <= charge_info["forced_charge_power"]
            else:
                return pyo.Constraint.Skip

        model.enforced_charge_power = pyo.Constraint(rule=enforced_charge_power_rule)

        def soc_charg_rule(model, t):
            if t in soc_imposed_i.keys():
                soc_i = soc_imposed_i[t]
            elif det[t] != 0:
                soc_i = model.soc_charg[t - 1]
            else:
                return pyo.Constraint.Skip

            return (
                model.soc_charg[t]
                == soc_i - model.charg_power[t] * eff / cap[t] * t_s_hours
            )

        model.soc_charg_rule = pyo.Constraint(model.time, rule=soc_charg_rule)

        def p_max_rule(model, t):
            if t == 0 and charge_is_enforced:
                return pyo.Constraint.Skip
            else:
                return -model.charg_power[t] <= p_max[t]

        model.p_max_rule = pyo.Constraint(model.time, rule=p_max_rule)

        def p_curve_rule(model, t):
            if p_max[t] == 0:
                return pyo.Constraint.Skip

            pow = [p_max[t], 1]
            soc = [0.9, 1.0]
            a, b = first_order_fit(soc, pow)
            # If there is no detention time or the low power is enforced, skip the constraint
            if det[t] == 0 or (t == 0 and charge_is_enforced):
                return pyo.Constraint.Skip

            return -model.charg_power[t] <= a * model.soc_charg[t] + b

        model.p_curve_rule = pyo.Constraint(model.time, rule=p_curve_rule)

    def build_obj(self):
        model = self.model

        def obj_cost_rule(model):
            return -model.price_obj

        model.obj = pyo.Objective(rule=obj_cost_rule, sense=pyo.minimize)

    def build_constr(
        self, pv_load, consumer_load, day_ahead, charge_info, relaxed=False
    ):
        self.set_grid_power_constr(consumer_load)

        self.set_pv_power(pv_load)

        # if len(self.microgrid.branches) == 1:
        #    self.set_branch_lim()

        self.trans_dis_constr()
        self.sum_price_cost_constr(day_ahead)

        self.soc_constr()
        self.charger_constr(charge_info)

        self.set_grid_peak_constr()


class NaiveMPCManager(SVLMPCManager):
    def __init__(self, microgrid, params_evaluation):
        super().__init__(microgrid, params_evaluation, forecast_mode="naive")


class ModelMPCManager(SVLMPCManager):
    def __init__(self, microgrid, params_evaluation):
        super().__init__(microgrid, params_evaluation, forecast_mode="model")


class ExperimentMPC(SVLMPCManager):
    def __init__(self, microgrid, params_evaluation):
        house_num = params_evaluation["house_num"]
        if not hasattr(self, "forecaster"):
            self.forecaster = ThreeMonthForecaster("model", house_num)
        super().__init__(microgrid, params_evaluation, forecast_mode="model")
        # If forecaster is not set, set it to the three month forecaster
        self.real_hist_peak = True
        self.ems_set = params_evaluation["ems_set"]

    def update_grid_peak(self):
        if len(self.microgrid.power_hist) == 0:
            prev_grid_power = 0
        else:
            prev_grid_power = self.microgrid.power_hist[0]["PublicGrid_0"][
                -1
            ].electrical

        if self.microgrid.datetime.hour == 15 and self.microgrid.datetime.minute == 0:
            self.update_hist_peak()

        elif prev_grid_power > self.current_month_peak:
            self.current_month_peak = prev_grid_power

    def update_hist_peak(self, real_import=False):
        hist_peak = self.current_month_peak
        env_hist = self.microgrid.env_hist[0]
        datetimes_env = env_hist["datetime"]
        ems_used = env_hist["ems_used"]
        success = env_hist["success"]
        if self.real_hist_peak:
            grid_import = env_hist["PublicGrid_0_import"]  # In kWh
        else:

            if len(self.microgrid.power_hist) == 0:
                datetimes_hist = []
                grid_power = []
            else:
                datetimes_hist = self.microgrid.power_hist[0]["datetime"]
                datetimes_hist = [dt.replace(tzinfo=None) for dt in datetimes_hist]
                grid_power = self.microgrid.power_hist[0]["PublicGrid_0"]
                grid_power = [p.electrical for p in grid_power]

        cur_time = self.microgrid.datetime.replace(tzinfo=None)
        limit_hist_check = cur_time.replace(hour=15, minute=0)

        max_hist_peak = 2.5
        for i, dt in enumerate(datetimes_env):
            dt = dt.replace(tzinfo=None)
            if dt >= limit_hist_check:
                break

            if ems_used[i] == self.ems_set and success[i] != "failed":
                if self.real_hist_peak:
                    ts_h = self.microgrid.time_step.total_seconds() / 3600
                    hist_peak = grid_import[i] / ts_h
                else:
                    index_grid = datetimes_hist.index(dt)
                    hist_peak = grid_power[index_grid]
                if hist_peak > max_hist_peak:
                    max_hist_peak = hist_peak
        self.current_month_peak = max_hist_peak


class SimulExpMPC(ExperimentMPC):
    def __init__(self, microgrid, params_evaluation):
        super().__init__(microgrid, params_evaluation)
        self.real_hist_peak = False


class PerfectExpMPC(ExperimentMPC):

    def __init__(self, microgrid, params_evaluation):
        SVLMPCManager.__init__(
            self, microgrid, params_evaluation, forecast_mode="perfect"
        )
        self.forecaster = ThreeMonthForecaster("perfect", self.house_num)
        self.real_hist_peak = False
        self.ems_set = params_evaluation["ems_set"]
        self.optimisation_interval = 1


class BugMPC(SimulExpMPC):
    def __init__(self, microgrid, params_evaluation):
        if not hasattr(self, "forecaster"):
            self.forecaster = BugForecaster("model", self.house_num)
        super().__init__(microgrid, params_evaluation)

    def set_hour_day_switch(self, hour):
        self.hour_day_switch = hour

    def prepare_info_constrs(self, retry_counter=0):
        pv_load, consumer_load, day_ahead, charge_info = super().prepare_info_constrs(
            retry_counter
        )

        cur_time = self.microgrid.utc_datetime
        if cur_time.hour == 15 and cur_time.minute == 0:
            charge_info["forced_charge_power"] = 0

        return pv_load, consumer_load, day_ahead, charge_info
