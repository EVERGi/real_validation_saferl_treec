import numpy as np

from simugrid.assets.asset import Asset, AssetType
from simugrid.simulation.definitions import EnergyVector
from simugrid.simulation.power import Power

import datetime


class Charger(Asset):
    """
    Create charger type asset

    :ivar name: Name of the charger
    :type name: str
    :ivar node: Parent node of the asset
    :type node: Node
    :ivar ID: id of the charger
    :type ID: int
    :ivar eff: efficiency of the charger
    :type eff: float
    :ivar max_charge_cp: Maximum power of the charger [kW]
    :type max_charge_cp: float
    :ivar max_discharge_cp: Maximum discharge power of the charger [kW]
    :type max_discharge_cp: float
    :ivar max_charge: Maximum power of the session [kW]
    :type max_charge: float
    :ivar max_discharge: Maximum discharge power of the session [kW]
    :type max_discharge: float
    :ivar soc: Current state of charge  of the battery [-]
    :type soc: float
    :ivar soc_f: Target state-of-charge [-]
    :type soc_f: float
    :ivar soc_min: Minimum state of charge of the battery [-]
    :type soc_min: float
    :ivar size: Capacity of the EV [kWh]
    :type size: float
    :ivar det: Parking time in timesteps [-]
    :type det: int
    :ivar prod_loss: Penalty cost if EV user preferences are not met
    :type prod_loss: float
    """

    def __init__(self, node, name):
        """
        Initialization of the Charger class

        :param node: the parent node of the asset
        :type node: Node
        :param name: name of the asset
        :type name: str
        """
        # Execture constructor of Asset class
        super().__init__(node, name)

        # Charge point (cp) information
        self.ID = None
        self.eff = 0.95
        self.max_charge_cp = 0
        self.max_discharge_cp = 0
        self.node = node  # Not sure this is useful
        self.kind = "smart"  # "smart" or "uncoord"

        # Charge session information
        self.max_charge = 0
        self.max_discharge = 0
        self.soc = 0
        self.soc_min = 0
        self.size = 0

        # User requirements
        self.soc_f = 0
        self.det = 0

        # Penalty cost if EV user preferences are not met
        self.prod_loss = 100

        # Specific to Julian EMS to normalize power
        self.max_production_power = -self.max_discharge_cp
        self.max_consumption_power = self.max_charge_cp

        # Specify type of asset
        self.energy_type = [EnergyVector.ELECTRIC]
        self.asset_type = AssetType.PROSUMER

        # Timestep
        self.dt_s = int(self.parent_node.microgrid.time_step.total_seconds())  # Seconds
        self.dt_h = self.dt_s / 3600  # Hours
        self.discrete_step = self.dt_h / self.dt_s

        # Bi-directional or not ?
        self.bi = False

    @property
    def environment_keys(self):
        return {
            "PowerProfile": [self.name],
            type(self).__name__: [
                f"p_max_{self.ID}",
                f"det_{self.ID}",
                f"soc_i_{self.ID}",
                f"soc_f_{self.ID}",
                f"capa_{self.ID}",
            ],
        }

    def set_attributes(self, var_dict):
        """
        Set asset attributes from dictionary

        :param var_dict: dict with attribute name as key and attribute value as value
        :type var_dict: dict
        """

        super().set_attributes(var_dict)

        # Specific to Julian EMS to normalize power
        self.max_production_power = -self.max_discharge_cp
        self.max_consumption_power = self.max_charge_cp

    def set_power_limits(self, environment):
        """
        Set the high and low power limits

        :param environment: Environment with information to set the power limits
        :type environment: Environment
        """

        # STEP 1: Update EV values

        # EV is connected to the charger, simply reduce det by 1 timestep
        if self.det > 1:
            # Reduce detention time
            self.det -= 1

            # No discharging
            if not self.bi:
                self.soc_min = self.soc

        # End of a charge session -> reset values
        else:
            self.det = 0
            self.max_discharge = 0
            self.max_charge = 0
            self.soc = 0
            self.soc_f = 0
            self.size = 0

        # Get environment information
        det_env = environment.env_values["det_" + str(self.ID)].value

        # New charging session ? and charger available ?
        if det_env != 0 and self.det == 0:
            # Electric vehicle information
            max_charge_ev = environment.env_values["p_max_" + str(self.ID)].value
            max_discharge_ev = 0
            if self.bi:
                max_discharge_ev = -10
            self.soc = environment.env_values["soc_i_" + str(self.ID)].value
            self.size = environment.env_values["capa_" + str(self.ID)].value

            # Set final EV info
            self.max_charge = min(self.max_charge_cp, max_charge_ev)
            self.max_discharge = max(self.max_discharge_cp, max_discharge_ev)

            # Save user requirements
            dt = self.parent_node.microgrid.time_step.total_seconds() / 3600
            self.det = int(np.round(det_env / dt))
            self.soc_min = self.soc
            if self.bi:
                self.soc_min = 0.2
            self.soc_f = environment.env_values["soc_f_" + str(self.ID)].value

        # STEP 2: Set power limits

        if self.det > 0:

            # Compute power bounds (from SOC bounds)
            charge, discharge = self.get_power_limits()

            # Set limits
            self.power_limit_low = Power(electrical=charge)
            self.power_limit_high = Power(electrical=discharge)

        # End of a charge session -> reset values
        else:
            self.power_limit_low = Power(electrical=0)
            self.power_limit_high = Power(electrical=0)

    def power_consequences(self):
        """
        Apply all consequences of the asset's power_output value

        :param reward: Reward to update based on the value of power_output
        :type reward: Reward
        """

        # Update soc (if an EV is connected)
        if self.det >= 1:
            if self.power_output.electrical < 0:
                add_soc = -1 * self.power_output.electrical * self.dt_h
                add_soc *= self.eff
                add_soc /= self.size
            else:
                add_soc = -1 * self.power_output.electrical * self.dt_h
                add_soc /= self.eff
                add_soc /= self.size

            self.soc = self.soc + add_soc

    # %%#############
    ### EV MODEL ###
    ################

    def get_current_state(self):
        state = {
            "det": self.det,
            "size": self.size,
            "soc": self.soc,
            "soc_f": self.soc_f,
            "soc_min": self.soc_min,
            "max_charge": self.max_charge,
            "max_discharge": self.max_discharge,
        }
        return state

    def get_power_limits(self):
        """
        Get SOC bounds and deduce Power bounds
        """
        # Collect state of EV
        state = self.get_current_state()

        soc_new, p_mean = self.ev_model(state, self.max_charge, self.soc)
        charge = float(-1 * p_mean)

        soc_new, p_mean = self.ev_model(state, self.max_discharge, self.soc)
        discharge = float(-1 * p_mean)

        # Update upper and lower SOC bounds
        self.bounds = self.battery_simulator(state)

        return (charge, discharge)

    def battery_simulator(self, state, buffer_time_h=0):
        """
        This function returns a list of the minimum SOC allowed
        on the time horizon of length 'detention'
        """
        #######################
        ##### Maximum soc #####
        #######################

        soc_t = state["soc"]
        soc_to_target_soc = [state["soc"]]

        t = 0
        while soc_t < state["soc_f"]:
            soc_t, p_mean = self.ev_model(state, state["max_charge"], soc_t)

            # Once close enough to target, stop charging
            if soc_t >= state["soc_f"]:
                soc_t = state["soc_f"]

            soc_to_target_soc.append(soc_t)

            # In case of an issue, raise exception
            t += 1
            if t > 300:
                raise Exception("Not possible to charge ev", soc_to_target_soc)

        # Full list of maximum SOC
        upper_soc = []
        for i in soc_to_target_soc:
            upper_soc.append(i)
        for _ in range(state["det"] + 1 - len(soc_to_target_soc)):
            upper_soc.append(state["soc_f"])

        ######################
        ##### Minimum SOC ####
        ######################

        # SOC to minimum soc
        soc_t = state["soc"]
        soc_to_soc_min = [state["soc"]]

        t = 0
        while soc_t > state["soc_min"]:
            soc_t, p_mean = self.ev_model(state, state["max_discharge"], soc_t)

            # Once close enough to target, stop discharging
            if soc_t <= state["soc_min"]:
                soc_t = state["soc_min"]

            soc_to_soc_min.append(soc_t)

            # In case of an issue, raise exception
            t += 1
            if t > 300:
                raise Exception("Not possible to charge ev", soc_to_soc_min)

        for _ in range(state["det"] + 1 - len(soc_to_soc_min)):
            soc_to_soc_min.append(state["soc_min"])
        # SOC min to SOC max
        soc_min_to_soc_target = self.get_lower_bound_end(buffer_time_h)

        for _ in range(state["det"] + 1 - len(soc_min_to_soc_target)):
            soc_min_to_soc_target.insert(0, state["soc_min"])

        # Remove timesteps if charging time too long
        soc_to_soc_min = soc_to_soc_min[: state["det"] + 1]
        soc_min_to_soc_target = soc_min_to_soc_target[-state["det"] - 1 :]

        # Concatenation of two previous lists
        lower_soc = []

        for j in range(1, state["det"] + 1 + 1):
            lower_soc.append(max(soc_min_to_soc_target[-j], soc_to_soc_min[-j]))

        # Reverse list
        lower_soc = lower_soc[::-1]

        # Annoying function
        for i in range(len(lower_soc)):
            if lower_soc[i] > upper_soc[i]:
                lower_soc[i] = upper_soc[i]

        return {"Lower SOC": lower_soc, "Upper SOC": upper_soc}

    def get_powers_to_reach_soc_final(self, buffer_time_h=0):
        state = self.get_current_state()
        soc_bounds = self.battery_simulator(state, buffer_time_h)
        if len(soc_bounds["Upper SOC"]) > 1:
            # Deduce minimum power for next timestep (charging)
            soc_diff = soc_bounds["Upper SOC"][1] - soc_bounds["Upper SOC"][0]

            if soc_diff > 0:
                power_soc_low = -1 * soc_diff * self.size / (self.dt_h * self.eff)
            else:
                power_soc_low = -1 * soc_diff * self.size / (self.dt_h / self.eff)

            # Deduce maximum power for next timestep (discharging)
            soc_diff = soc_bounds["Lower SOC"][1] - soc_bounds["Lower SOC"][0]
            if soc_diff > 0:
                power_soc_high = -1 * soc_diff * self.size / (self.dt_h * self.eff)
            else:
                power_soc_high = -1 * soc_diff * self.size / (self.dt_h / self.eff)

            power_limit_low, power_limit_high = self.get_power_limits()

            power_low = max(power_limit_low, power_soc_low)
            power_high = min(power_limit_high, power_soc_high)
        else:
            power_low, power_high = self.get_power_limits()

        return power_low, power_high

    def get_lower_bound_end(self, buffer_time_h=0):
        """
        This function returns a list of the minimum SOC per time-step
        to reach the final SOC at the end of the detention time.
        A buffer time can be added to finnish charging earlier.

        :param buffer_time_h: time in hours to finish charging earlier
        :type buffer_time_h: int

        :return: lower_bound_end: list of minimum SOC allowed
        """
        state = self.get_current_state()

        soc_min = state["soc_min"]
        soc_f = state["soc_f"]
        if soc_f <= soc_min:
            lower_bound_end = [soc_min]
            return lower_bound_end

        charge_time_h = self.get_charge_time_h(state, soc_min, soc_f)
        charge_time_h += buffer_time_h

        num_full_charge = int(charge_time_h // self.dt_h)
        time_first_charge_h = charge_time_h % self.dt_h

        max_charge = state["max_charge"]
        soc = soc_min
        lower_bound_end = [soc]
        soc = self.calc_charge_soc(state, max_charge, soc, time_first_charge_h)
        lower_bound_end.append(soc)

        for _ in range(num_full_charge):
            soc = self.calc_charge_soc(state, max_charge, soc, self.dt_h)
            lower_bound_end.append(soc)

        return lower_bound_end

    def get_charge_time_h(self, state, soc_init, soc_f):
        # Defining necessary variable for the ev model
        eff = self.eff
        capacity = state["size"]
        max_charge = state["max_charge"]
        power = max_charge
        # Maximum possible power when the battery is almost fully charged
        low_max_charge = 1.0
        # Soc at which the charge curve starts when charging at maximum possible power
        soc_max_charge_tip = 0.9
        soc_tipping = soc_max_charge_tip

        _, get_time = charging_functions(
            soc_init,
            eff,
            capacity,
            power,
            soc_tipping,
            soc_max_charge_tip,
            max_charge,
            low_max_charge,
        )

        time_to_soc_f = get_time(soc_f)

        return time_to_soc_f

    def calc_charge_soc(self, state, power, soc_init, time_step_h):
        # Defining necessary variable for the ev model
        soc_f = state["soc_f"]
        eff = self.eff
        capacity = state["size"]
        max_charge = state["max_charge"]
        # Maximum possible power when the battery is almost fully charged
        low_max_charge = 1.0
        # Soc at which the charge curve starts when charging at maximum possible power
        soc_max_charge_tip = 0.9

        final_soc, _ = calc_charge_soc_and_power(
            soc_init,
            soc_f,
            eff,
            power,
            capacity,
            max_charge,
            time_step_h,
            low_max_charge,
            soc_max_charge_tip,
        )

        return final_soc

    def ev_model(self, state, power, soc_init):
        """
        Simulating charging/discharging in one timestep
        Output the SOC(t+1) and the average power in the timestep
        """
        # Defining necessary variable for the ev model
        soc_f = state["soc_f"]
        eff = self.eff
        capacity = state["size"]
        max_charge = state["max_charge"]
        # Time_step in hours
        time_step_h = self.dt_h
        # Maximum possible power when the battery is almost fully charged
        low_max_charge = 1.0
        # Soc at which the charge curve starts when charging at maximum possible power
        soc_max_charge_tip = 0.9
        soc_min = state["soc_min"]

        # Calculate the soc and power at the end of the time_step for charging and discharging
        if power > 0:
            final_soc, average_power = calc_charge_soc_and_power(
                soc_init,
                soc_f,
                eff,
                power,
                capacity,
                max_charge,
                time_step_h,
                low_max_charge,
                soc_max_charge_tip,
            )
        elif power < 0:
            final_soc, average_power = calc_discharge_soc_and_power(
                soc_init, eff, power, capacity, time_step_h, soc_min
            )
        else:
            final_soc = soc_init
            average_power = 0

        return final_soc, average_power

    def milp_arrays(self, environment):
        """
        Create arrays of values needed for MILP asset.
        Return arrays for soc_max, soc_min, capacity, max_charge_power
        and max_discharge_power
        """
        start_dt = self.parent_node.microgrid.start_time
        end_dt = self.parent_node.microgrid.end_time
        time_step = self.parent_node.microgrid.time_step.total_seconds() / 3600

        p_max = np.array(
            environment.env_values["p_max_" + str(self.ID)].sample_range(
                start_dt, end_dt
            )["values"]
        )

        det = np.array(
            environment.env_values["det_" + str(self.ID)].sample_range(
                start_dt, end_dt
            )["values"]
        ).astype(float)
        det = (det / time_step).astype(int)

        soc_i = np.array(
            environment.env_values["soc_i_" + str(self.ID)].sample_range(
                start_dt, end_dt
            )["values"]
        )
        soc_f = np.array(
            environment.env_values["soc_f_" + str(self.ID)].sample_range(
                start_dt, end_dt
            )["values"]
        )
        capa = np.array(
            environment.env_values["capa_" + str(self.ID)].sample_range(
                start_dt, end_dt
            )["values"]
        )

        self.soc_min = np.zeros(len(soc_i))
        self.soc_max = np.zeros(len(soc_i))
        self.capacity = np.zeros(len(soc_i))
        self.max_charge_rate = np.zeros(len(soc_i))
        self.max_discharge_rate = np.zeros(len(soc_i))

        for i in range(len(p_max)):
            if det[i] != 0:
                self.capacity[i : i + det[i] + 1 : 1] = capa[i]
                self.max_charge_rate[i : i + det[i] : 1] = (
                    min(self.max_charge_cp, p_max[i]) / capa[i]
                )
                self.max_discharge_rate[i : i + det[i] : 1] = (
                    min(self.max_discharge_cp, p_max[i]) / capa[i]
                )
                max_charge = min(self.max_charge_cp, p_max[i])
                max_discharge = min(self.max_discharge_cp, p_max[i])
                state = {
                    "det": det[i],
                    "size": capa[i],
                    "soc": soc_i[i],
                    "soc_f": soc_f[i],
                    "soc_min": soc_i[i],
                    "max_charge": max_charge,
                    "max_discharge": max_discharge,
                }
                soc_dict = self.battery_simulator(state)

                if self.kind == "smart":
                    # print("smart")
                    self.soc_min[i : i + det[i] + 1 : 1] = np.array(
                        soc_dict["Lower SOC"]
                    )
                    self.soc_max[i : i + det[i] + 1 : 1] = np.array(
                        soc_dict["Upper SOC"]
                    )
                else:
                    # print("uncoord")
                    self.soc_min[i : i + det[i] + 1 : 1] = np.array(
                        soc_dict["Upper SOC"]
                    )
                    self.soc_max[i : i + det[i] + 1 : 1] = np.array(
                        soc_dict["Upper SOC"]
                    )
            else:
                continue
        self.det = det
        return (
            self.soc_min,
            self.soc_max,
            self.capacity,
            self.max_charge_rate,
            self.max_discharge_rate,
            self.det,
        )


def calc_charge_soc_and_power(
    soc_init,
    soc_f,
    eff,
    power,
    capacity,
    max_charge,
    time_step,
    low_max_charge,
    soc_max_charge_tip,
):
    SOC_FULL = 1.0
    # Points for linear relation between soc and charging power
    x = [max_charge, low_max_charge]
    y = [soc_max_charge_tip, SOC_FULL]

    # Find tipping SOC in function of power input via characteristics of EV
    if power <= low_max_charge:
        soc_tipping = SOC_FULL
    elif power >= max_charge:
        soc_tipping = soc_max_charge_tip
    else:
        coeff_tipping = first_order_fit(x, y)
        soc_tipping = coeff_tipping[0] * power + coeff_tipping[1]

    # Get function that returns soc of the battery after charging for a given time-step
    get_soc, _ = charging_functions(
        soc_init,
        eff,
        capacity,
        power,
        soc_tipping,
        soc_max_charge_tip,
        max_charge,
        low_max_charge,
    )
    final_soc = get_soc(time_step)

    if final_soc > soc_f:
        final_soc = soc_f

    if final_soc == soc_init:
        average_power = 0
    else:
        average_power = (final_soc - soc_init) * capacity / (eff * time_step)

    return final_soc, average_power


def calc_discharge_soc_and_power(soc_init, eff, power, capacity, time_step, soc_min):
    final_soc = soc_init + ((1 / eff) * power * time_step) / capacity

    if final_soc < soc_min:
        final_soc = soc_min

    average_power = (final_soc - soc_init) * capacity / (time_step / eff)

    return final_soc, average_power


def first_order_fit(x, y):
    """
    Fit a first order polynomial to the data
    """
    slope = (y[1] - y[0]) / (x[1] - x[0])
    intercept = y[0] - slope * x[0]
    return [slope, intercept]


def charging_functions(
    soc_init,
    efficiency,
    capacity,
    power,
    soc_tipping,
    soc_max_charge_tip,
    max_charge,
    low_max_charge,
):
    SOC_FULL = 1.0
    # Calculate the slope between soc and time when the soc is lower than soc_tipping is no charging curve
    slope_charge = power * efficiency / capacity

    # Calculate the 3 variables that define the charging curve when soc is higher than soc_tipping
    a, b, c = get_soc_curve_params(
        max_charge, efficiency, capacity, low_max_charge, soc_max_charge_tip
    )
    # Time in the charging curve time frame at which the power tips from normal charging to curve charging
    curve_time_start = time_curve(soc_tipping, a, b, c)

    # Time in the charging curve time frame at which the power charging ends
    curve_time_end = time_curve(SOC_FULL, a, b, c)

    # Calculate the time in the simulation timeframe at which the charging behaviour tips
    # and soc at the current time if the ev only did normal charging (intercept_soc)
    if soc_init <= soc_tipping:
        # In this case the soc at current time on the normal charging function is simply the initial soc
        intercept_soc = soc_init
        # The behaviour tipping time is calculated with using the normal charging function
        time_at_tipping = (soc_tipping - intercept_soc) / slope_charge
    else:
        # In this case the start of the behaviour tipping happened before the current time
        # so this behaviour tipping is calculated using the charging curve
        curve_time_soc_init = time_curve(soc_init, a, b, c)
        time_at_tipping = -(curve_time_soc_init - curve_time_start)
        # The soc at current time on the normal charging function is in this case the soc that would have occured
        # if the ev would have continued charging using the normal charging behaviour and not the chaging curve
        intercept_soc = soc_init - slope_charge * time_at_tipping

    # End of the charging time in the simulation timeframe
    charge_time_end = time_at_tipping + (curve_time_end - curve_time_start)

    # Function to calculate soc given the simulation time
    def calc_soc(time):
        if time <= time_at_tipping:
            soc = slope_charge * time + intercept_soc
        elif time_at_tipping < time <= charge_time_end:
            # Switch from the simulation timeframe to the curve timeframe
            curve_time = time - time_at_tipping + curve_time_start
            soc = soc_curve(curve_time, a, b, c)
        else:
            soc = SOC_FULL
        return soc

    # Function to calculate simulation time given the soc
    def calc_time(soc):
        if soc <= soc_tipping:
            time = (soc - intercept_soc) / slope_charge
        elif soc_tipping < soc <= SOC_FULL:
            curve_time = time_curve(soc, a, b, c)
            # Switch from the curve timeframe to the simulation timeframe
            time = curve_time - curve_time_start + time_at_tipping
        else:
            time = charge_time_end
        return time

    return calc_soc, calc_time


def soc_curve(time, a, b, c):
    return a * np.exp(c * time) + b


def time_curve(soc, a, b, c):
    return np.log((soc - b) / a) / c


def get_soc_curve_params(max_charge, eff, capacity, low_max_charge, soc_max_charge_tip):
    SOC_FULL = 1
    # Points for linear relation between soc and charging power
    s = [soc_max_charge_tip, SOC_FULL]
    p = [max_charge, low_max_charge]

    # Get the relation between SOC and power
    # s is for soc and p for power
    # Relation is p=a*s+b
    coeff_tipping = first_order_fit(s, p)
    a = coeff_tipping[0]
    b = coeff_tipping[1]

    # Following is the simplification of relation between soc and time
    # The base equation calculates the difference in soc ds related to the difference in time dt
    # ds = (eff/capacity)p*dt
    # ds = (eff/capacity)(a*s+b)*dt
    # 1/(a*s+b)ds = (eff/capacity)*dt
    # Now we integrate both sides
    # ln(a*s+b)/a = (eff/capacity)*t+c
    # ln(a*s+b) = a*(eff/capacity)*t+c
    # a*s+b = exp(a*(eff/capacity)*t+c)
    # s = (exp(a*(eff/capacity)*t+c)-b)/a
    # To calculate c we use the fact that at t=0 s=soc_max_charge_tip
    # soc_max_charge_tip = (exp(a*(eff/capacity)*0+c)-b)/a
    # soc_max_charge_tip = (exp(c)-b)/a
    # exp(c) can be simplified to c
    # soc_max_charge_tip = (c-b)/a
    # c = soc_max_charge_tip*a+b

    eff_cap = eff / capacity

    constant_t = a * soc_max_charge_tip + b

    new_a = constant_t / a
    new_b = -b / a
    new_c = a * eff_cap

    # soc = new_a * np.exp(time*new_c) + new_b

    return new_a, new_b, new_c


# %%
