import numpy as np
import os
from .variants_handler import VariantsHandler as vh


DATA_VARIANTS = "./data/data_variants.csv"


class Constant:
    def __init__(self, variant, save_type, save_path):
        self.PATH_TO_DIRECTORY = save_path
        # self.PATH_TO_PLOTS = save_path + "/PLOTS"
        self.PATH_TO_RESULTS = save_path + "/RESULTS"
        # self.PATH_TO_PLOTS_PGF = save_path + "/PLOTS_PGF"
        self.TYPE_NAMES = save_type  # default is ['png', 'pgf'] for saving. In GUI use custom combination pgf or png
        self.DATA_TABLE_NAME = "./data/dataCSV.csv"

        variable = vh(DATA_VARIANTS, variant)
        self.M_OGR = variable.M_OGR
        self.V_I_MAX = variable.V_I_MAX
        self.M0 = variable.M0
        self.OTN_M_TSN = variable.OTN_M_TSN
        self.OTN_M_T = variable.OTN_M_T
        self.OTN_M_CH = variable.OTN_M_CH
        self.OTN_P_0 = variable.OTN_P_0
        self.CE_0 = variable.CE_0
        self.N_DV = variable.N_DV
        self.N_REV = variable.N_REV
        self.PS = variable.PS
        self.B_A = variable.B_A
        self.OTN_L_GO = variable.OTN_L_GO

        self.G = 9.81
        self.MACH = np.array(
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        )
        self.Hk = 11
        self.H_step = 1
        self.H = np.arange(stop=self.Hk+self.H_step, step=self.H_step, dtype="f")
        self.OTN_M = 0.95

        self.OTN_M_T_SNP = 0.015
        self.OTN_M_T_ANZ = 0.05
        self.OTN_M_T_PR = 0.01
        self.OTN_M_EMPTY = 1 - self.OTN_M_T - self.OTN_M_TSN

        # Data for take-off and landing part
        self.ALPHA_TO = 2
        self.ALPHA_LA = 2

        self.ALPHA_RT = 6
        self.ALPHA_TD = 6

        self.H_TO = 10.7
        self.H_LA = 15

        self.F_TO = 0.02
        self.F_LA = 0.2

        # Data for turn calculation part
        self.TURN_M_MIN = 0.4
        self.TURN_M_MAX = 0.8
        self.TURN_H = 6
        self.TURN_n_ye = 3

        # Data for control and stability part
        self.OTN_S_GO = np.array([0.01, 0.2])
        self.SIGMA_N_MIN = -0.1
        self.FI_MAX = -25
        self.FI_UST = -4
        self.MZ0_BGO_LA = -0.03
        self.DELTA_OTN_X_E = 0.15
