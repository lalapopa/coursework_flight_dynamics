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
    def get_row_name_table_1():
        return np.array(
                [ 
                    "$M$",
                    "$V$",
                    "$V$",
                    "$q$",
                    "$C_{y_n}$",
                    "$K_n$",
                    "$P_n*10^{-5}$",
                    "$P_p*10^{-5}$",
                    r"$\Delta \bar{p}(n_x)$",
                    "$V_y^*$",
                    r"$\bar{R}_{кр}$",
                    "$q_{ч}$",
                    "$q_{км}$",
                ],
        )

    @staticmethod
    def get_row_units_table_1():
        return np.array(
                [ 
                    r"$-$",
                    r"$\frac{м}{с}$",
                    r"$\frac{км}{ч}$",
                    r"$\frac{H}{м^2}$",
                    r"$-$",
                    r"$-$",
                    r"$H$",
                    r"$H$",
                    r"$-$",
                    r"$\frac{м}{с}$",
                    r"$-$",
                    r"$\frac{кг}{ч}$",
                    r"$\frac{кг}{км}$",
                ],
        )

    @staticmethod
    def get_row_name_table_2():
        return np.array(
                [
                    "H",
                    "Vy*_max",
                    "M_min_dop",
                    "M_max_dop",
                    "M_min",
                    "M_max",
                    "M_1",
                    "M_2",
                    "V_3",
                    "V_4",
                    "M_4",
                    "q_ch_min",
                    "q_km_min",
                ]
        )

    @staticmethod
    def get_row_name_table_2_latex():
        return np.array(
                [
                    r"$H$",
                    r"$V_{y_{max}}^*$",
                    r"$\underset{\min \, доп}{M [V]}$",
                    r"$\underset{\max \, доп}{M [V]}$",
                    r"$\underset{\min}{M [V]}$",
                    r"$\underset{\max}{M [V]}$",
                    r"$\underset{(P_п\, min)}{M_1 [V_1]}$",
                    r"$\underset{(V_{y_{max}}^*)}{M_2 [V_2]}$",
                    r"$\underset{(q_{ч_{\min}})}{V_3}$",
                    r"$\underset{(q_{{км}_{\min}})}{V_4}$",
                    r"$M_4$",
                    r"$q_{ч_{\min}}$",
                    r"$q_{{км}_{\min}}$",
                ]
        )

    @staticmethod
    def get_row_units_table_2_latex():
        return np.array(
                [
                    r"$км$",
                    r"$\frac{м}{с}$",
                    r"$-\,[\frac{км}{ч}]$",
                    r"$-\,[\frac{км}{ч}]$",
                    r"$-\,[\frac{км}{ч}]$",
                    r"$-\,[\frac{км}{ч}]$",
                    r"$-\,[\frac{км}{ч}]$",
                    r"$-\,[\frac{км}{ч}]$",
                    r"$\frac{км}{ч}$",
                    r"$\frac{км}{ч}$",
                    r"$-$",
                    r"$\frac{кг}{ч}$",
                    r"$\frac{кг}{км}$",
                ]
        )
        

    @staticmethod
    def get_row_name_table_3(descent_or_climb="climb"):
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
                    "H",
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
        )

    @staticmethod
    def get_row_name_table_3_tex(descent_or_climb="climb"):
        condition = ("climb", "descent")
        if descent_or_climb not in condition:
            raise InvalidNameError(descent_or_climb)
        if descent_or_climb == "climb":
            M_text = r"$\underset{наб}{M}$"
            Vy_text = r"$V_{y_{наб}}$"
            L_text = r"$L_{наб}$"
            t_text = r"$t_{наб}$"
            mass_fuel_text = r"$m_{T_{наб}}$"
        if descent_or_climb == "descent":
            M_text = r"$\underset{сн}{M}$"
            Vy_text = r"$V_{y_{сн}}$"
            L_text = r"$L_{сн}$"
            t_text = r"$t_{сн}$"
            mass_fuel_text = r"$m_{T_{сн}}$"
        return np.array(
                [
                    [
                        r"$\underset{узел}{H}$",
                        M_text,
                        r"$V$",
                        r"$V_{км}$",
                        r"$\frac{\Delta V}{\Delta H}$",
                        r"$n_x$",
                        r"$V_{y}^*$",
                        r"$\theta$",
                        Vy_text,
                        r"$H_э$",
                        r"$\Delta H_э$",
                        r"$n_{x_{ср}}$",
                        r"$\frac{\Delta H_{э}}{1000 n_x}$",
                        ],[
                            r"$P$",
                            r"$\frac{CeP}{V_y^*}$",
                            r"$(\frac{CeP}{V_y^*})_{ср}$",
                            r"$\frac{\Delta H_э}{3600}(\frac{CeP}{V_y^*})_{ср}$",
                            L_text,
                            r"$V_{y_{ср}}^*$",
                            t_text,
                            r"$Ce$",
                            ],
                        ], dtype=object
        )
    @staticmethod
    def get_row_units_table_3_latex():
        return np.array([[
            r'м', 
            '-',
            r'$\frac{м}{с}$',
            r'$\frac{км}{ч}$',
            r'$\frac{1}{с}$',
            r'-', 
            r'$\frac{м}{с}$',
            r'$град.$',
            r'$\frac{м}{с}$',
            r'м', 
            r'м', 
            r'-', 
            r'км',
            ], [ 
            r'$H$',
            r'-',
            r'-',
            r'кг',
            r'км',
            r'$\frac{м}{с}$',
            r'мин',
            r'$\frac{кг}{H ч}$',]
            ], dtype=object)

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
    def get_mini_table_3_latex(descent_or_climb="climb"):
        condition = ("climb", "descent")
        if descent_or_climb not in condition:
            raise InvalidNameError(descent_or_climb)
        if descent_or_climb == "climb":
            return np.array(["$m_{T_{наб}}$", "$L_{наб}$", "$t_{наб}$"])
        if descent_or_climb == "descent":
            return np.array(["$m_{T_{сн}}$", "$L_{сн}$", "$t_{сн}$"])

    @staticmethod
    def get_mini_table_3_units():
        return np.array(['Кг', 'Км', 'Мин'], dtype=object)


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
    def get_kr_table_latex():
        return np.array(
            [
                r"$T_{кр}$",
                r"$L_{кр}$",
                r"$\rho_{H\, кр}$",
                r"$H_{0\, кр}$",
                r"$H_{к\, кр}$",
            ]
        )

    @staticmethod
    def get_kr_units_table():
        return np.array(
            [
                r"мин",
                r"км",
                r"$\frac{кг}{м^3}$",
                r"км",
                r"км",
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
    def get_to_la_table_latex():
        return np.array(
            [
                r"$V_{отр}$",
                r"$L_{р}$",
                r"$L_{вд}$",
                r"$V_{кас}$",
                r"$L_{пр}$",
                r"$L_{пд}$",
            ]
        )

    @staticmethod
    def get_to_la_units_table():
        return np.array(
            [
                r"$\frac{м}{с}$",
                r"$м$",
                r"$м$",
                r"$\frac{м}{с}$",
                r"$м$",
                r"$м$",
            ]
        )

    @staticmethod
    def get_row_name_turn_table_part_1(latex=False):
        if latex:
            return np.array([
                r"$M$",
                r"$V$",
                r"$V$",
                r"$q$",
                r"$C_{y_{ГП}}$",
                r"$C_{y_{доп}}$",
                r"$n_{y_{доп}}$",
                r"$K_{ГП}$",
                r"$P_{n}*10^{-5}$",
                r"$P_{p}*10^{-5}$",
                ])
        else:
            return np.array(
                [
                    "M",
                    "V [m/s]",
                    "V [km/h]",
                    "q [H/m^2]",
                    "Cy_gp",
                    "Cy_dop",
                    "n_y_dop",
                    "K_gp",
                    "P_potr",
                    "P_rasp",
                ],
        )

    @staticmethod
    def get_row_name_turn_table_part_2(latex=False):
        if latex:
            return np.array([
                r"$\bar{P}$",
                r"$n_{y_{p}}$",
                r"$n_{y_{вир}}$",
                r"$\omega_{вир}$",
                r"$r_{вир}$",
                r"$t_{вир}$",
                ])
        else:
            return np.array(
                [
                    "otn_P",
                    "n_y_p",
                    "n_y_vir",
                    "omega_vir",
                    "radius_vir [m]",
                    "time_vir [s]",
                ],
        )
    @staticmethod
    def get_row_name_turn_table_part_1_units():
        return np.array([
            r"-",
            r"$\frac{м}{с}$",
            r"$\frac{км}{ч}$",
            r"$\frac{H}{м^2}$",
            r"-",
            r"-",
            r"-",
            r"-",
            r"H",
            r"H",
            ])

    @staticmethod
    def get_row_name_turn_table_part_2_units():
        return np.array([
            r"-",
            r"-",
            r"-",
            r"$\frac{1}{с}$",
            r"$м$",
            r"$с$",
            ])

    @staticmethod
    def get_row_name_otn_S_go():
        return np.array(
            ["otn_S_go", "otn_x_tpp", "otn_x_tpz"],
            dtype=object,
        )

    @staticmethod
    def get_row_name_phis_table():
        return np.array(
                [
                    "M",
                    "V",
                    "phi_bal",
                    "phi_n",
                    "n_y_p",
                ],
        )

    def get_row_name_sigmas():
        return np.array(
                [
                    "M",
                    "otn_x_F",
                    "otn_x_H",
                    "otn_x_TPZ",
                    "sigma_n",
                ],
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

    @staticmethod
    def get_row_cargo_table_latex():
        return np.array(
                        ["Режим", "$L$", "$m_{цн}$"],
                        dtype=object,
                        )

    @staticmethod
    def get_row_units_cargo_table():
        return np.array([
            r"№",
            r"км",
            r"кг",
            ],dtype=object)

class InvalidNameError(TypeError):
    pass
