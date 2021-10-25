import numpy as np


class TextHandler:
    def __init__(self, alt):
        self.alt = alt
        self.values = {
            "P": {
                "label": "$P [H]$",
                "label_in_box": [
                    "$P_р(H={}[Км])$".format(self.alt),
                    "$P_п(H={}[Км])$".format(self.alt),
                ],
                "plot_text": ["$M_{min P}=$", "$M_{max P}=$", "$M(P_{пmin})=$"],
            },
            "M": {
                "label": "$M$",
                "label_in_box": "",
                "plot_text": "",
            },
            "H": {
                "label": "$H [Км]$",
                "label_in_box": "",
                "plot_text": ["$H_{ст}=$", "$H_{пр}=$"],
            },
            "V": {
                "label": "$V [м/с^2]$",
                "label_in_box": "",
                "plot_text": "",
            },
            "C_y": {
                "label": "$C_y$",
                "label_in_box": [
                    "$C_{y_{доп}}$",
                    "$C_y(H={}[Км])$".format(self.alt),
                ],
                "plot_text": ["$M_{min_{доп}}=$"],
            },
            "V_y": {
                "label": "$V_y^* [м/с^2]$",
                "label_in_box": [
                    "$V_y^*(H={}[Км])$".format(self.alt),
                    "$V_{y_{max}}^*(H)$",
                ],
                "plot_text": ["$V_{y_{max}}^*=$"],
            },
            "q_ch": {
                "label": [
                    "$q_ч [Кг/ч]$",
                    "$q_ч [т/ч]$",
                ],
                "label_in_box": [
                    "$q_ч(H={}[Км])$".format(self.alt),
                    "$q_{ч_{min}}(H)$",
                ],
                "plot_text": ["$V(q_{ч_{min}})=$"],
            },
            "q_km": {
                "label": "$q_{км} [кг/км]$",
                "label_in_box": [
                    "$q_{{км}}(H={}[Км])$".format(self.alt),
                    "$q_{км_{min}}(H)$",
                ],
                "plot_text": ["$V(q_{км_{min}})=$"],
            },
        }

    def get_label(self, name_value):
        return self.values[name_value]["label"]

    def get_label_in_box(self, name_value):
        return self.values[name_value]["label_in_box"]

    def get_plot_text(self, name_value):
        return self.values[name_value]["plot_text"]

    def get_all_name(self):
        return [label for label in self.values]

    @staticmethod
    def get_row_name_table_1(MACH):
        return np.array(
            [
                [
                    "V",
                    "V_km_h",
                    "q",
                    "C_y_n",
                    "K_n",
                    "P_potr*10^-5",
                    "P_rasp*10^-5",
                    "n_x",
                    "V_y",
                    "otn_R",
                    "q_ch",
                    "q_km",
                ],
                MACH,
            ],
            dtype=object,
        )

    @staticmethod
    def get_row_name_table_2(alts):
        return np.array(
            [
                [
                    "Vy*_max",
                    "M_min_dop",
                    "M_max_dop",
                    "M_min",
                    "M_max",
                    "M_1",
                    "M_2",
                    "V_3",
                    "V_4",
                    "q_ch_min",
                    "q_km_min",
                ],
                alts,
            ],
            dtype=object,
        )

    @staticmethod
    def get_row_name_table_3(alts, descent_or_climb="climb"):
        condition = ("climb", "descent")
        if descent_or_climb not in condition:
            raise InvalidNameError(descent_or_climb)
        if descent_or_climb == "climb":
            M_text = "M_nab"
            Vy_text = "Vy_nab"
            L_text = "L_nab"
            t_text = "t_nab"
            mass_fuel_text = "m_T_nab"
        if descent_or_climb == "descent":
            M_text = "M_des"
            Vy_text = "Vy_des"
            L_text = "L_des"
            t_text = "t_des"
            mass_fuel_text = "m_T_des"
        return np.array(
            [
                [
                    M_text,
                    "V",
                    "V_km",
                    "deltaV/deltaH",
                    "n_x",
                    "Vy*",
                    "teta",
                    Vy_text,
                    "He",
                    "deltaHe",
                    "n_x_avg",
                    "deltaHe/1000*n_x",
                    "P",
                    "CeP/Vy*",
                    "(CeP/Vy*)avg",
                    "deltaHe/3600*(CeP/Vy*)avg",
                    L_text,
                    "Vy*avg",
                    t_text,
                    "Ce",
                ],
                alts,
            ],
            dtype=object,
        )

    @staticmethod
    def get_mini_table_3(descent_or_climb="climb"):
        condition = ("climb", "descent")
        if descent_or_climb not in condition:
            raise InvalidNameError(descent_or_climb)
        if descent_or_climb == "climb":
            return np.array(["m_t_nab [kg]", "L_nab [km]", "t_nab [min]"])
        if descent_or_climb == "descent":
            return np.array(["m_t_des [kg]", "L_des [km]", "t_des [min]"])

    @staticmethod
    def get_kr_table():
        return np.array(
            [
                "T_kr [min]",
                "L_kr [km]",
                "Ro [kg/m^3]",
                "H_0kr [km]",
                "H_kkr [km]",
            ]
        )

    @staticmethod
    def get_to_la_table():
        return np.array(
            [
                "V_otr [m/s^2]",
                "L_raz [m]",
                "L_vd [m]",
                "V_kas [m/s^2]",
                "L_pr [m]",
                "L_pd [m]",
            ]
        )

    @staticmethod
    def get_row_name_turn_table(mach):
        return np.array(
            [
                [
                    "otn_P",
                    "n_y_p",
                    "n_y_vir",
                    "omega_vir",
                    "radius_vir [m]",
                    "time_vir [s]",
                ],
                mach,
            ],
            dtype=object,
        )

    @staticmethod
    def get_row_name_otn_S_go():
        return np.array(
            ["otn_S_go", "otn_x_tpp", "otn_x_tpz"],
            dtype=object,
        )

    @staticmethod
    def get_row_name_phis_table(mach):
        return np.array(
            [
                [
                    "V",
                    "phi_bal",
                    "phi_n",
                    "n_y_p",
                ],
                mach,
            ],
            dtype=object,
        )

    def get_row_name_sigmas(mach):
        return np.array(
            [
                [
                    "otn_x_F",
                    "otn_x_H",
                    "otn_x_TPZ",
                    "sigma_n",
                ],
                mach,
            ],
            dtype=object,
        )

    @staticmethod
    def get_row_cargo_table():
        return np.array(
            [
                ["L [km]", "m_cargo [kg]"],
                [
                    "1",
                    "2",
                    "3",
                ],
            ],
            dtype=object,
        )


class InvalidNameError(TypeError):
    pass
