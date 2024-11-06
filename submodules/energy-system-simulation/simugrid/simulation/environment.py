import datetime
import pandas as pd
import sys
import inspect
from copy import deepcopy
import numpy as np

import csv


class Environment:
    """
    Environment of a node
    Keeps track of the environmental values of a node that are
    independent of the actions taken by the energy management system
    :ivar env_values: Dictionary with all environmental values
    :type env_values: dict
    :ivar micrgrid: microgrid the environment is part of
    :type microgrid: Microgrid
    :ivar nodes: List of nodes that use data of the environment
    :type nodes: List[Node]
    """

    def __init__(self, microgrid, nodes=None, env_folder_path=None):
        self.env_values = dict()
        self.microgrid = microgrid
        microgrid.environments += [self]

        self.env_folder_path = env_folder_path
        num_dt = int((microgrid.end_time - microgrid.start_time) / microgrid.time_step)
        self.time = [
            microgrid.start_time + i * microgrid.time_step for i in range(num_dt)
        ]

        self.time_index = list(range(len(self.time)))

        if nodes is not None:
            for node in nodes:
                node.set_environment(self)

    def update(self):
        """
        Update the env_values of the environment
        """
        for _, value in self.env_values.items():
            try:
                value.sample_next()
            except ValueError:
                value.sample_date()

    def get_index(self):
        for i, environment in enumerate(self.microgrid.environments):
            if environment == self:
                return i

    @property
    def index(self):
        return self.get_index()

    def get_time(self):
        return self.time

    def get_time_index(self):
        return self.time_index

    def add_value(self, value_name, value):
        """
        adds a constant value to the environment

        :param value_name: the name of the value
        :type value_name: str
        :param value: the value
        :type value: any
        """
        if type(value) in (int, float):
            const_value = ConstValue(self.microgrid, value)
            self.env_values[value_name] = const_value
        elif type(value) is list:
            list_value = ListValue(self.microgrid, value)
            self.env_values[value_name] = list_value
        elif callable(value):
            func_value = FuncValue(self.microgrid, value)
            self.env_values[value_name] = func_value
        elif type(value) is str and value[-4:] == ".csv":
            csv_list, _ = CsvValue.csv_to_lists(value)
            csv_value = CsvValue(self.microgrid, csv_list[0])

            self.env_values[value_name] = csv_value

    def add_function_values(self, function_file):
        """
        adds a function file to the environment. Each function in this
        function will be added to the environment with
        the name of the function the reference in the environment file

        :param function_file: the path of the file containing all the function
        :type function_file: str
        """
        filename = function_file.split("/")[-1]
        module_name = filename.split(".")[0]
        function_folder = function_file.replace(filename, "")
        sys.path.insert(1, function_folder)
        __import__(module_name)
        functions = inspect.getmembers(sys.modules[module_name], inspect.isfunction)
        for function in functions:
            self.add_value(function[0], function[1])

    def add_multicolumn_csv_values(self, filename):
        """
        adds a csv file to the environment. The csv file must contain headers and
            each column must have the same length
        The first column must be the column containing the timestamps. All the
            others columns will be added to the
        environment with the name of the column the reference
            in the environment (case sensitive)

        The timestamps should be one of following formats:
        ['%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M', '%Y-%m-%d %H:%M:%S']

        :param filename: the path of the csv file
        :type filename: str

        """
        if type(filename) is str and filename[-4:] == ".csv":
            all_csv_lists, header = CsvValue.csv_to_lists(filename)
            for i, csv_list in enumerate(all_csv_lists):
                value_name = header[i]
                csv_value = CsvValue(self.microgrid, csv_list)
                self.env_values[value_name] = csv_value

    @classmethod
    def environment_from_json(cls, json, microgrid):
        """
        Create environment from json

        :param json: the object in json form
        :type json: dict
        :param microgrid: the microgrid
        :type microgrid: Microgrid

        :return: created environment
        :rtype: Environment
        """

        values = json["values"]
        environment = cls(microgrid=microgrid)

        for value in values:
            if value["type"] == "value":
                name = value["name"]
                environment.add_value(name, value["value"])
            elif value["type"] == "csv":
                environment.add_multicolumn_csv_values(value["name"])

        return environment


class Value:
    """
    A help class to structure information the environmental values of the microgrid

    :ivar microgrid: the microgrid to which the environmental microgrid belongs
    :type microgrid: Microgrid
    :ivar value: the value of a environment parameter
        for the current datetime of the microgrid
    :type value: any
    """

    def __init__(self, microgrid):
        self.microgrid = microgrid
        self.value = None

    def sample_next(self):
        """
        Samples the values for the next timestamp of the microgrid
        """
        pass

    def reset(self):
        """
        Resets the value to its initial value
        """
        pass

    def sample_range(self, start_dt, end_dt):
        """
        Generates a 2d numpy array of values between the start datetime
            and the end datetime included
        with the datetimes in one column and the corresponding values in the other.
        """
        pass

    def get_forecast(
        self,
        start_time,
        end_time,
        quality="perfect",
        naive_back=datetime.timedelta(days=1),
    ):
        forecast = dict()

        num_dt = int((end_time - start_time) / self.microgrid.time_step)
        datetime_range = [
            start_time + i * self.microgrid.time_step for i in range(num_dt)
        ]

        if quality == "naive":
            start_time -= naive_back
            end_time -= naive_back

        time_sample = self.get_forec_time_samples(start_time, end_time)

        forecast = {"datetime": datetime_range, "values": list()}
        for times in time_sample:
            sample = self.sample_range(times[0], times[1])
            forecast["values"] += list(sample["values"])
        return forecast

    def get_forec_time_samples(self, start_time, end_time):
        microgrid = self.microgrid
        time_diff = end_time - start_time

        diff_start_time = microgrid.start_time - start_time
        diff_end_time = end_time - microgrid.end_time

        if start_time < microgrid.start_time and end_time <= microgrid.start_time:
            start_time = microgrid.end_time - diff_start_time
            end_time = start_time + time_diff
            return [[start_time, end_time]]
        elif start_time >= microgrid.end_time and end_time > microgrid.end_time:
            end_time = microgrid.start_time + diff_end_time
            start_time = end_time - time_diff
            return [[start_time, end_time]]

        if start_time < microgrid.start_time:
            start_time = microgrid.end_time - diff_start_time
            return [[start_time, microgrid.end_time], [microgrid.start_time, end_time]]
        elif end_time > microgrid.end_time:
            end_time = microgrid.start_time + diff_end_time
            return [[start_time, microgrid.end_time], [microgrid.start_time, end_time]]

        return [[start_time, end_time]]


class ConstValue(Value):
    """
    A class that inheritance from the Value class.
    The value will in this case be a constant

    :ivar microgrid: the microgrid to which the environmental microgrid belongs
    :type microgrid: Microgrid
    :ivar value: the value of a environment parameter
        for the current datetime of the microgrid
    :type value: float
    """

    def __init__(self, microgrid, value):
        Value.__init__(self, microgrid)
        self.const_value = value

    def sample_next(self):
        """
        Samples the values for the next timestamp of the microgrid
        """
        self.value = self.const_value

    def sample_range(self, start_dt, end_dt):
        """
        Generates a 2d numpy array of values between the start datetime
            and the end datetime included
        with the datetimes in one column and the corresponding values in the other.
        """

        num_dt = int((end_dt - start_dt) / self.microgrid.time_step)
        date_range = [start_dt + i * self.microgrid.time_step for i in range(num_dt)]

        values = np.array([self.const_value] * num_dt)
        return {"datetime": date_range, "values": list(values)}


class FuncValue(Value):
    """
    A class that inheritance from the Value class.
    The value will in this case come from a function

    :ivar microgrid: the microgrid to which the environmental microgrid belongs
    :type microgrid: Microgrid
    :ivar func_value: the value of a environment parameter for the current
        datetime of the microgrid
    :type func_value: callable
    """

    def __init__(self, microgrid, func_value):
        Value.__init__(self, microgrid)
        self.value_func = func_value

    def sample_next(self):
        """
        Samples the values for the next timestamp of the microgrid
        """
        self.value = self.value_func(self.microgrid, self.microgrid.datetime)

    def sample_range(self, start_dt, end_dt):
        """
        Generates a 2d numpy array of values between the start datetime and
            the end datetime included
        with the datetimes in one column and the corresponding values in the other.
        """
        num_dt = int((end_dt - start_dt) / self.microgrid.time_step)
        date_range = [start_dt + i * self.microgrid.time_step for i in num_dt]

        values = np.array([self.value_func(self.microgrid, dt) for dt in date_range])

        return {"datetime": date_range, "values": list(values)}


class CsvValue(Value):
    """Environment based on csv files
    Environment object that extracts info for each environmental
    variables from csv files.
    """

    def __init__(self, microgrid, csv_list):
        """
        # TODO: this function is not documented, Julian

        """
        Value.__init__(self, microgrid)

        self.microgrid = microgrid

        # and associated files to read from
        self.cur_index = 0

        # Dictionary with all environmental variables
        # and associated values
        self.value = None

        self.csv_list = csv_list

    @staticmethod
    def csv_to_lists(csv_path):
        """
        # TODO: this function is not documented, Julian

        """

        with open(csv_path) as csvfile:
            csvreader = csv.reader(csvfile)
            header_row = next(csvreader)
            header = [head.split(" ")[0] for head in header_row[1:]]

            final_list = [[] for _ in header]
            for row in csvreader:
                datetime_row = None
                formats = [
                    "%d/%m/%Y %H:%M:%S",
                    "%d/%m/%Y %H:%M",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%dT%H:%M:%SZ",
                ]
                for form in formats:
                    try:
                        datetime_row = datetime.datetime.strptime(row[0], form)
                    except ValueError:
                        pass

                # If no format found, raise error
                if datetime_row is None:
                    raise ValueError(
                        "time format of {} is not supported.please \
                        change all the dates to one of following "
                        "formats: {} ".format(csv_path, formats)
                    )

                for i, value in enumerate(row[1:]):
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    final_list[i].append([datetime_row, value])
        return final_list, header

    def sample_date(self):
        """
        # TODO: this function is not documented, Julian

        """
        value = None
        microgrid = self.microgrid
        naive_datetime = microgrid.datetime.replace(tzinfo=None)
        for i in range(len(self.csv_list)):
            if self.csv_list[i][0] == naive_datetime:
                value = self.csv_list[i][1]
                break

        self.cur_index = i + 1

        self.value = value

    def sample_next(self):
        """
        # TODO: this function is not documented, Julian

        """
        if self.value is None:
            self.sample_date()
        microgrid = self.microgrid
        cur_ind = self.cur_index
        naive_datetime = microgrid.datetime.replace(tzinfo=None)
        pd_datetime = pd.Timestamp(naive_datetime, tz=None)

        if pd_datetime == self.csv_list[cur_ind][0]:
            value = self.csv_list[cur_ind][1]

        else:
            raise ValueError(
                "Datetime in csv file does not correspond to next expected datetime."
                "Check timezone or error in csv environment file."
            )

        self.cur_index += 1

        self.value = value

    def sample_range(self, start_dt, end_dt):
        """
        # TODO: this function is not documented, Julian

        """
        time_step = self.microgrid.time_step
        num_dt = int((end_dt - start_dt) / time_step)
        date_range = [start_dt + i * time_step for i in range(num_dt)]

        first_date = self.csv_list[0][0]
        localized_first = self.microgrid.local_tz.localize(first_date)
        if start_dt.tzinfo is None:
            localized_start = self.microgrid.local_tz.localize(start_dt)
        else:
            localized_start = start_dt

        index_start = int((localized_start - localized_first) / time_step)
        index_end = index_start + num_dt
        values = np.array([i[1] for i in self.csv_list[index_start:index_end]])

        return {"datetime": date_range, "values": values}

    def __deepcopy__(self, memo):
        """ """
        new_csv_value = CsvValue(deepcopy(self.microgrid, memo), self.csv_list)
        new_csv_value.cur_index = self.cur_index
        new_csv_value.value = self.value

        return new_csv_value


class SeriesValue(Value):
    """
    Environment based on pandas series

    """

    def __init__(self, microgrid, series_list):
        """
        :param microgrid: the microgrid object
        :type microgrid: Microgrid
        :param series_list: pandas series
        :type series_list: pandas.Series
        """

        Value.__init__(self, microgrid)

        self.cur_index = 0

        # Dictionary with all environmental variables
        # and associated values
        self.value = None
        self.name = series_list.name
        dates = series_list.index.values.tolist()
        dates = map(lambda x: datetime.datetime.utcfromtimestamp(x / 1e9), dates)
        self.list = list(zip(dates, series_list.to_list()))

    def sample_date(self):
        """
        sample the value at the current time

        """
        value = None
        microgrid = self.microgrid
        naive_datetime = microgrid.datetime.replace(tzinfo=None)
        for i in range(len(self.list)):
            if self.list[i][0] == naive_datetime:
                value = self.list[i][1]
                break

        self.cur_index = i + 1

        self.value = value

    def sample_next(self):
        """
        samples the next value in the list

        """
        if self.value is None:
            self.sample_date()
        microgrid = self.microgrid
        cur_ind = self.cur_index
        naive_datetime = microgrid.datetime.replace(tzinfo=None)
        pd_datetime = pd.Timestamp(naive_datetime, tz=None)

        if pd_datetime == self.list[cur_ind][0]:
            value = self.list[cur_ind][1]

        else:
            raise ValueError(
                "Datetime in series does not correspond to next expected datetime."
                "Check timezone or error in csv environment file."
            )

        self.cur_index += 1

        self.value = value

    def sample_range(self, start_dt, end_dt):
        """
        sample values in range

        :param start_dt: start date
        :type start_dt: datetime.datetime
        :param end_dt: end date
        :type end_dt: datetime.datetime

        :return: dictionary with datetime and values
        :rtype: dict
        """
        time_step = self.microgrid.time_step
        num_dt = int((end_dt - start_dt) / time_step)
        date_range = [start_dt + i * time_step for i in range(num_dt)]

        first_date = self.list[0][0]
        localized_first = self.microgrid.local_tz.localize(first_date)
        if start_dt.tzinfo is None:
            localized_start = self.microgrid.local_tz.localize(start_dt)
        else:
            localized_start = start_dt

        index_start = int((localized_start - localized_first) / time_step)
        index_end = index_start + num_dt
        values = np.array([i[1] for i in self.list[index_start:index_end]])

        return {"datetime": date_range, "values": values}


class ListValue(Value):
    """
    A class that inheritance from the Value class.
    The value will in this case be a constant

    :ivar microgrid: the microgrid to which the environmental microgrid belongs
    :type microgrid: Microgrid
    :ivar value: the value of a environment parameter for the current datetime of the microgrid
    :type value: float
    """

    def __init__(self, microgrid, value):
        Value.__init__(self, microgrid)
        self.list_values = value

    def sample_next(self):
        """
        Samples the values for the next timestamp of the microgrid
        """
        self.value = self.list_values[
            int(
                (self.microgrid.datetime - self.microgrid.start_time)
                / self.microgrid.time_step
            )
        ]

    def sample_range(self, start_dt, end_dt):
        """
        Generates a 2d numpy array of values between the start datetime and the end datetime included
        with the datetimes in one column and the corresponding values in the other.
        """
        date_range = pd.date_range(
            start_dt,
            end_dt,
            freq=self.microgrid.time_step,
            tz=self.microgrid.local_tz,
            inclusive="left",
        )
        values = self.list_values
        return {"datetime": date_range, "values": values}
