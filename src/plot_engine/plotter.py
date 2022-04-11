import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

sys.path.insert(0, "..")
from DataHandler import DataHandler as dh


class Plotter:
    def __init__(self, x, save_type, **kwargs):

        self.x_values = x
        self.y_values = [array for array in kwargs.values()]
        self.save_type = save_type
        Plotter.setup_matplotlib(self.save_type)

    def get_figure(self, *legend_labels, t_graph=False):
        plt.figure(figsize=(6, 4.5))
        if t_graph:
            plot = lambda x, y, legend: self.add_plot(y, x, legend)
        else:
            plot = lambda x, y, legend: self.add_plot(x, y, legend)

        for y_value, legend_label in zip(self.y_values, legend_labels):
            plot(self.x_values, y_value, legend_label)
        plt.grid()

    def add_plot(self, x, y, label):
        plt.plot(x, y, label=label)

    def add_labels(self, x_label, y_label):
        plt.xlabel(x_label)
        plt.ylabel(y_label)

    def add_text(self, x, y, positions, *texts, add_value="x", text_location="up", marker_style='o'):
        plt.plot(x[positions], y[positions], marker_style)
        if add_value == "x":
            prefix_value = lambda pos: str(round(x[pos], 3))
        elif add_value == "y":
            prefix_value = lambda pos: str(round(y[pos], 3))
        else:
            prefix_value = lambda pos: ""

        if text_location == "up":
            position_sign = 1
        elif text_location == "down":
            position_sign = -1
        else:
            position_sign = 0

        try:
            for pos, text in zip(positions, texts):
                self._render_text(
                    text,
                    x[pos],
                    y[pos] + position_sign * Plotter.two_percent_of_axis(y),
                    prefix_value(pos),
                )
        except TypeError:
            self._render_text(
                texts[0],
                x[positions],
                y[positions],
                prefix_value(positions),
            )
        return

    def _render_text(self, text, x_pos, y_pos, adding_value=""):
        plt.annotate(text + adding_value, xy=(x_pos, y_pos))

    @staticmethod
    def two_percent_of_axis(x):
        full_size = np.amax(x)
        return (2 * full_size) / 100

    def show_plot(self):
        plt.show()

    def set_legend(self):
        plt.legend()

    def save_figure(self, name, path):
        self.path = path
        if self.save_type == "png" or "default":
            self._save_as_png(name)
        if self.save_type == "pgf":
            self._save_as_pgf(name)
        print(f"Saved plot: '{name}.{self.save_type}'")

    def _save_as_png(self, name):
        dh.change_dir(self.path + "/PLOTS_PNG")
        plt.savefig("{}.png".format(name), dpi=300)
        dh.change_dir(go_back=True)

    def _save_as_pgf(self, name):
        dh.change_dir(self.path + "/PLOTS_PGF")
        plt.savefig("{}.pgf".format(name))
        dh.change_dir(go_back=True)

    def close_plot(self):
        plt.cla()
        plt.close()

    def set_notation(self, power):
        ax = plt.gca()
        y_formatter = ScalarFormatter(useOffset=True, useMathText=True)
        y_formatter.set_powerlimits((-2, power))
        ax.yaxis.set_major_formatter(y_formatter)

    @staticmethod
    def setup_matplotlib(type_name):
        if type_name == "png":
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            matplotlib.use("AGG")
            return True
        if type_name == "pgf":
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            matplotlib.use(type_name)
            matplotlib.rcParams.update(
                {
                    "pgf.texsystem": "pdflatex",
                    "font.family": "serif",
                    "text.usetex": True,
                    "pgf.rcfonts": False,
                    "pgf.preamble": "\n".join(
                        [
                            r"\usepackage[warn]{mathtext}",
                            r"\usepackage[T2A]{fontenc}",
                            r"\usepackage[utf8]{inputenc}",
                            r"\usepackage[english,russian]{babel}",
                        ]
                    ),
                }
            )
            return True
        if type_name == "default":
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            matplotlib.use("TkAgg")
            return True
