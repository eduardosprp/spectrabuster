from datetime import time, date, datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import numpy as np
import os


class Saver(object):

    """
    Class used to take the logic of storing files away from the Spectrum class and the
    main script. Unlike Spectrum, it makes no checks whatsoever, so the likelihood of
    whatever you're doing going wrong if you don't fully understand this code is quite high.
    """

    def __init__(self, prefix, path="./"):
        # {{{
        self.prefix = prefix
        self.path = path

        self.today = date.today()
        self.today_string = self.today.strftime("%d-%m-%Y")

        self.full_path = f"{path}{prefix}.dat"

        self.x = []
        self.y = []

        if (
            os.path.isfile(self.full_path)
            and Saver._get_file_date(self.full_path) == self.today_string
        ):
            self.x, self.y, _ = Saver._read_file(self.full_path, self.today)
        else:
            date_comment = f"# date = {self.today_string}"
            Saver._write_value(self.full_path, "w", date_comment)

    # }}}

    def add(self, *value):
        # {{{
        """
        Adds a value to the file. Creates a new file if the day has changed. Also
        stores an array of the values saved that day, in order to plot the cummulative
        graph. If value is a tuple, it will create as many columns as there are elements
        in the tuple.
        """

        now = datetime.now()
        now_string = now.strftime("%H:%M:%S")
        self.today = date.today()
        self.today_string = self.today.strftime("%d-%m-%Y")

        file_date = Saver._get_file_date(self.full_path)

        if self.today_string == file_date:
            self.x.append(now)
            self.y.append(value)

            Saver._write_value(self.full_path, "a", now_string, *value)

        else:
            self.today = date.today()
            self.today_string = date.today().strftime("%d-%m-%Y")

            self.x = [now]
            self.y = [value]

            date_comment = f"# date = {self.today_string}"
            Saver._write_value(self.full_path, "w", date_comment)

    # }}}

    @staticmethod
    def _write_value(path, mode, col1_val, *values):
        # {{{
        """
        Internal method to write values to a file in a columns format.
        Will write as many columns as there are values.
        """

        with open(path, mode) as arq:
            arq.write(f"{col1_val}")

            for value in values:
                arq.write(f"\t{value}")

            arq.write(f"\n")

    # }}}

    def save_plot(self, style="-", title="", xlabel=None, ylabel=None, prefix=None):
        # {{{
        """
        Stores the plot of the data collected up to now.
        """

        if prefix is None:
            prefix = self.prefix

        fig, ax = plt.subplots()

        # Code I don't understand that magically formats the x axis
        locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlim([self.today, self.today + timedelta(days=1)])

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title} {self.today_string}")

        ax.plot_date(self.x, self.y, style)

        self.fig_full_path = f"{self.path}{prefix}.png"

        fig.savefig(self.fig_full_path)
        plt.close(fig)

    # }}}

    @staticmethod
    def _read_file(path, today):
        # {{{

        """
        Reads a file and return the values in its columns as arrays. Used
        to set the initial values of self.x and self.y.
        """

        dict_args = {}
        col1 = []
        col2 = []
        year, month, day = today.year, today.month, today.day

        with open(path, "r") as arq:

            gen_lines = (line.split() for line in arq)

            for line_split in gen_lines:
                if line_split[0] == "#":
                    dict_args[line_split[1]] = line_split[3]
                else:
                    hour, minute, second = line_split[0].split(":")
                    hour, minute, second = int(hour), int(minute), int(second)

                    datetime_tuple = (year, month, day, hour, minute, second)
                    time_value = datetime(*datetime_tuple)

                    col1.append(time_value)

                    other_cols = tuple(
                        Saver._try_convert(i, None) for i in line_split[1:]
                    )

                    col2.append(other_cols)

        return col1, col2, dict_args

    # }}}

    @staticmethod
    def _try_convert(value, default):
        # {{{
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    # }}}

    @staticmethod
    def _get_file_date(path):
        # {{{
        try:
            with open(path, "r") as arq:
                file_date = arq.readline().split()[-1]
            return file_date
        except IndexError:
            return None

    # }}}

    def backup(self, path):
        today = datetime.today().strftime("%d-%m-%Y")
        file_name = f"{path}-{today}"
        os.system(f"cp {self.full_path} {file_name}.dat")
        os.system(f"cp {self.fig_full_path} {file-name}.png")

    # TODO: implement (more clever) backups
