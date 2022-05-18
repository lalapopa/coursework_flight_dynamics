from DataHandler import DataHandler as dh
import numpy as np


class VariantsHandler:
    def __init__(self, file_name, variant):
        df = dh(file_name)
        self.M_OGR = float(df.get_column("M_OGR", "variant", np.array([variant])))
        self.V_I_MAX = float(df.get_column("V_i", "variant", np.array([variant])))
        self.M0 = float(df.get_column("m0", "variant", np.array([variant])) * 1000)
        self.OTN_M_TSN = float(
            df.get_column("otn_m_tsn", "variant", np.array([variant]))
        )
        self.OTN_M_T = float(df.get_column("otn_m_t", "variant", np.array([variant])))
        self.OTN_M_CH = float(df.get_column("otn_m_CH", "variant", np.array([variant])))
        self.OTN_P_0 = float(df.get_column("otn_p_0", "variant", np.array([variant])))
        self.CE_0 = float(df.get_column("Ce0", "variant", np.array([variant])))
        self.N_DV = float(df.get_column("n_dv", "variant", np.array([variant])))
        self.N_REV = float(df.get_column("n_rev", "variant", np.array([variant])))
        self.PS = float(df.get_column("ps", "variant", np.array([variant])))
        self.B_A = float(df.get_column("ba", "variant", np.array([variant])))
        self.OTN_L_GO = float(df.get_column("otn_L_go", "variant", np.array([variant])))
        self.S = float(df.get_column("S", "variant", np.array([variant])))

