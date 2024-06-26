import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from scipy import interpolate

from DataHandler import DataHandler as dh
from .data import Constant as cnst
from .calc import Formulas as frmls
from .text_engine import TextHandler as text_handler
from .plot_creating_using_data import PlotBuilderUsingData as pbud


def main(
    var: int,
    save_type: list[str],
    save_folder: str,
    step_size: float = 1,
):
    global const

    const = cnst(var, save_type, save_folder)
    calc = Calculation()
    H_static, H_practical = find_celling(calc)
    const.H = dh.proper_array(0, H_practical, step_size)

    for alt in const.H:
        if alt <= 11:
            alt = alt.astype(float)
        calc.first_part(alt, save_plot=True, save_data=True)
    calc.second_part(const.H, H_practical, H_static, save_plot=True, save_data=True)
    calc.climb_part(save_plot=True)
    calc.level_flight_part(debug=False)
    calc.descent_part(save_plot=True)
    calc.cargo_possibility(save_plot=True)
    calc.take_off_landing_part()
    calc.turn_part(save_plot=True)
    calc.static_stability_control_part(H_practical, H_static, save_plot=True)
    print("DONE :)")


class Calculation:
    def __init__(self):
        self.df = dh(const.DATA_TABLE_NAME)
        self.__define_variables()

    def first_part(self, altitude, save_plot=False, save_data=False):
        self.altitude = altitude
        DATA = self.run_calculation_part_one(altitude)

        self.run_plot_first_part(run_save=save_plot)
        if save_data:
            dh.save_data(
                DATA,
                text_handler.get_row_name_table_1(),
                f"alt_{str(self.altitude)}.csv",
                const.PATH_TO_RESULTS,
            )
            DATA_tex = []
            precision_order = [2, 0, 0, 0, 3, 2, 3, 3, 3, 1, 2, 0, 2]
            for j, array in enumerate(DATA):
                formated_array = []
                for i, value in enumerate(array):
                    if i in const.MACH_output_index:
                        formated_array.append(f"%.{precision_order[j]}f" % value)
                DATA_tex.append(formated_array)

            dh.save_data_tex(
                DATA_tex,
                text_handler.get_row_name_table_1(),
                f"alt_{str(self.altitude)}.tex",
                const.PATH_TO_RESULTS,
                units_value=text_handler.get_row_units_table_1(),
            )

    def run_calculation_part_one(self, altitude):
        self.altitude = round(altitude, 4)
        self.take_constant()
        return self.calculate_using_formulas()

    def get_V_y(self, altitude):
        self.altitude = altitude
        self.take_constant()
        DATA = self.calculate_using_formulas()
        return DATA[9]

    def second_part(self, alts, H_pr, H_st, save_plot=False, save_data=False):
        self.altitude = alts
        self.H_pr = H_pr
        self.H_st = H_st
        self.M_max_dop = self.find_M_max_dop(self.altitude)
        self.M_min = self.find_M_min()
        self.M_max = self.find_M_max()
        DATA = self.prepare_data_for_H_tab()

        if save_data:
            dh.save_data(
                DATA,
                text_handler.get_row_name_table_2(),
                "table_2.csv",
                const.PATH_TO_RESULTS,
            )
            DATA_tex = self.prepare_data_for_H_tab_latex()
            dh.save_data_tex(
                DATA_tex,
                text_handler.get_row_name_table_2_latex(),
                "table_2.tex",
                const.PATH_TO_RESULTS,
                units_value=text_handler.get_row_units_table_2_latex(),
            )
        if save_plot:
            self.run_plot_second_part()

    def prepare_data_for_H_tab(self):
        return np.array(
            [
                self.altitude,
                self.Vy_max,
                self.M_min_dop,
                self.M_max_dop,
                self.M_min,
                self.M_max,
                self.M_1,
                self.M_2,
                self.V_3,
                self.V_4,
                self.M_4,
                self.q_ch_min,
                self.q_km_min,
            ],
            dtype=object,
        )

    def prepare_data_for_H_tab_latex(self):
        a_H = self.df.get_column(
            "a_H", "H", np.array([self.altitude]), inter_value=True
        )
        V_km_func = lambda V, a_sos: V * a_sos * 3.6

        M_min_dop_column = np.array(
            [
                f"$%.3f\, [%.0f]$" % (value, V_km_func(value, a_H[i]))
                for i, value in enumerate(self.M_min_dop)
            ]
        )

        M_max_dop_column = np.array(
            [
                f"$%.3f\, [%.0f]$" % (value, V_km_func(value, a_H[i]))
                for i, value in enumerate(self.M_max_dop)
            ]
        )
        M_min_column = np.array(
            [
                f"$%.3f\, [%.0f]$" % (value, V_km_func(value, a_H[i]))
                for i, value in enumerate(self.M_min)
            ]
        )

        M_max_column = np.array(
            [
                f"$%.3f\, [%.0f]$" % (value, V_km_func(value, a_H[i]))
                for i, value in enumerate(self.M_max)
            ]
        )
        M_1_column = np.array(
            [
                f"$%.3f\, [%.0f]$" % (value, V_km_func(value, a_H[i]))
                for i, value in enumerate(self.M_1)
            ]
        )
        M_2_column = np.array(
            [
                f"$%.3f\, [%.0f]$" % (value, V_km_func(value, a_H[i]))
                for i, value in enumerate(self.M_2)
            ]
        )
        return np.array(
            [
                [str(round(val, 2)) for val in self.altitude],
                [str(round(val, 2)) for val in self.Vy_max],
                M_min_dop_column,
                M_max_dop_column,
                M_min_column,
                M_max_column,
                M_1_column,
                M_2_column,
                [f"%.0f" % val for val in self.V_3],
                [f"%.0f" % val for val in self.V_4],
                [f"%.3f" % val for val in self.M_4],
                [str(round(val, 2)) for val in self.q_ch_min],
                [str(round(val, 2)) for val in self.q_km_min],
            ]
        )

    def take_constant(self):
        self.tilda_P = self.find_tilda_P(const.MACH, self.altitude)
        self.tilda_Ce = self.find_Ce_tilda(const.MACH, self.altitude)
        self.a_H = self.df.get_column(
            "a_H", "H", np.array([self.altitude]), inter_value=True
        )
        self.Ro_H = self.df.get_column(
            "Ro_H", "H", np.array([self.altitude]), inter_value=True
        )

        self.Cy_dop = self.df.get_column(
            "Cydop", "M", const.MACH, extend_row=True, inter_value=True
        )
        self.C_x_m = self.df.get_column(
            "Cxm", "M", const.MACH, extend_row=True, inter_value=True
        )
        self.A = self.df.get_column(
            "A", "M", const.MACH, extend_row=True, inter_value=True
        )
        self.C_y_m = self.df.get_column(
            "Cym", "M", const.MACH, extend_row=True, inter_value=True
        )

        self.p_h = self.df.get_column(
            "P_H", "H", np.array([self.altitude]), inter_value=True
        )
        self.p_h_11 = self.df.get_column("P_H", "H", np.array([11]))

    def find_tilda_P(self, f_M, f_H):
        H = np.append(np.arange(start=0, stop=11, step=2), 11)
        M = self.df.get_column("M_H_ptilda")
        P = np.array([self.df.get_column(f"Ptilda{alt}") for alt in H])
        tilda_P = interpolate.interp2d(M, H, P, kind="linear")
        return tilda_P(f_M, f_H)

    def find_Ce_tilda(self, f_M, f_H):
        H = np.append(np.arange(start=0, stop=11, step=2), 11)
        M = self.df.get_column("M_H_ce")
        Ce = np.array([self.df.get_column(f"Cetilda{alt}") for alt in H])
        tilda_Ce = interpolate.interp2d(M, H, Ce, kind="linear")
        return tilda_Ce(f_M, f_H)

    def __what_alt_take(self):
        if self.altitude >= 11:
            return str(11)
        else:
            return str(self.altitude)

    def calculate_using_formulas(self):
        self.V = frmls.v_speed(const.MACH, self.a_H)
        self.V_km_h = self.V * 3.6
        self.q = frmls.q_dynamic_pressure(self.V, self.Ro_H)
        self.C_y_n = frmls.C_y_n_lift_coefficient(const.OTN_M, const.PS, self.q)
        self.C_x_n = frmls.C_x_n_drag_coefficient(
            self.C_x_m,
            self.A,
            self.C_y_n,
            self.C_y_m,
        )
        self.K_n = frmls.K_n_lift_to_drag_ratio(self.C_y_n, self.C_x_n)
        self.P_potr = frmls.P_potr_equation(const.OTN_M, const.M0, const.G, self.K_n)
        self.P_rasp = frmls.P_rasp_equation(
            const.OTN_P_0,
            const.M0,
            const.G,
            self.tilda_P,
            self.altitude,
            self.p_h_11,
            self.p_h,
        )
        self.n_x = frmls.n_x_equation(
            const.OTN_M,
            const.M0,
            const.G,
            self.P_rasp,
            self.P_potr,
        )
        self.P_des = self.P_rasp * 0.06
        self.n_x_des = frmls.n_x_equation(
            const.OTN_M,
            const.M0,
            const.G,
            self.P_des,
            self.P_potr,
        )

        self.V_y = frmls.V_y_equation(self.V, self.n_x)
        self.otn_R = frmls.otn_R_equation(self.P_rasp, self.P_potr)
        self.Cedr = self.find_Cedr_depends_R(self.otn_R)

        self.Ce = frmls.q_ch_hour_consumption(
            const.CE_0,
            self.tilda_Ce,
            1,
            1,
        )
        self.Ce_des = frmls.q_ch_hour_consumption(
            1.91 * const.CE_0,
            self.tilda_Ce,
            1,
            1,
        )
        self.q_ch = frmls.q_ch_hour_consumption(
            const.CE_0,
            self.tilda_Ce,
            self.Cedr,
            self.P_potr,
        )

        self.q_km = frmls.q_km_range_consumption(self.q_ch, self.V)

        self.q_ch_flying, self.V_flying = self.find_flying_area(
            self.otn_R, self.V, self.q_ch
        )
        self.q_km_flying, self.V_flying = self.find_flying_area(
            self.otn_R, self.V, self.q_km
        )
        self.M_flying = self.V_flying / self.a_H

        return np.array(
            [
                const.MACH,
                self.V,
                self.V_km_h,
                self.q,
                self.C_y_n,
                self.K_n,
                self.P_potr / 100000,
                self.P_rasp / 100000,
                self.n_x,
                self.V_y,
                self.otn_R,
                self.q_ch,
                self.q_km,
            ]
        )

    def run_plot_first_part(self, run_save=False):
        build_plot = pbud(
            self.altitude, const.MACH, const.TYPE_NAMES, const.PATH_TO_DIRECTORY
        )
        MminP, MmaxP, M_1 = build_plot.plot_P(self.P_potr, self.P_rasp, save=run_save)
        M_min_dop = build_plot.plot_C_y_C_dop(self.C_y_n, self.Cy_dop, save=run_save)
        M_2, Vy_max = build_plot.plot_V_y(self.V_y, save=run_save)

        self.M_min_P = np.append(self.M_min_P, MminP)
        self.M_max_P = np.append(self.M_max_P, MmaxP)
        self.M_1 = np.append(self.M_1, M_1)
        self.M_min_dop = np.append(self.M_min_dop, M_min_dop)
        self.M_2 = np.append(self.M_2, M_2)
        self.Vy_max = np.append(self.Vy_max, Vy_max)

        self.find_fuel_consumption(build_plot, run_save=run_save)

    def find_fuel_consumption(self, plot_builder, run_save=False):
        self.M_max_dop = self.find_M_max_dop(self.altitude)
        M_max = self.find_M_max()
        try:
            V_3, q_ch_min = plot_builder.plot_q_ch(
                self.V_flying, self.q_ch_flying, save=run_save
            )
            V_4, q_km_min = plot_builder.plot_q_km(
                self.V_flying, self.q_km_flying, save=run_save
            )
        except ValueError:
            V_3, q_ch_min = plot_builder.plot_q_ch(self.V, self.q_ch, save=run_save)
            V_4, q_km_min = plot_builder.plot_q_km(self.V, self.q_km, save=run_save)

        M_4 = V_4 / self.a_H
        self.M_4 = np.append(self.M_4, M_4)
        self.V_3 = np.append(self.V_3, V_3)
        self.V_4 = np.append(self.V_4, V_4)
        self.q_ch_min = np.append(self.q_ch_min, q_ch_min)
        self.q_km_min = np.append(self.q_km_min, q_km_min)
        plot_builder.plot_q_km_q_ch_together(
            self.q_km_flying,
            self.q_ch_flying,
            self.V_flying,
            q_km_min,
            q_ch_min,
            V_4,
            V_3,
            f"q_km_ch_together_H={round(self.altitude, 1)}",
            save=run_save,
        )

    def find_Cedr_depends_R(self, otn_R):
        Rdr_interp, Cedr_interp = self.df.interp_two_column("Rdr", "Cedr")
        Rdr_position = dh.get_index_nearest_element_in_array(Rdr_interp, otn_R)
        return Cedr_interp[Rdr_position - 1]

    def find_flying_area(self, R_values, V_speed, value):
        accept_position = np.nonzero(R_values < 1)
        return value[accept_position], V_speed[accept_position]

    def find_M_max_dop(self, alts):
        alts = np.unique(np.array([alts]))
        sqrt_delta_minus = self.df.get_column(
            "square_delta", "H", alts, inter_value=True
        )
        self.M_Vi_max = frmls.M_V_i_max(const.V_I_MAX, sqrt_delta_minus, self.a_H)
        self.M_OGR = np.array([const.M_OGR for i in alts])
        return dh.find_min_max_from_arrays(self.M_OGR, self.M_Vi_max)

    def find_M_min(self):
        return dh.find_min_max_from_arrays(self.M_min_dop, self.M_min_P, find_min=False)

    def find_M_max(self):
        return dh.find_min_max_from_arrays(self.M_max_dop, self.M_max_P, self.M_OGR)

    def run_plot_second_part(self, run_save=True):
        build_plot = pbud(
            self.altitude, const.MACH, const.TYPE_NAMES, const.PATH_TO_DIRECTORY
        )
        build_plot.plot_V_y_H(
            self.Vy_max, self.altitude, self.H_pr, self.H_st, save=run_save
        )
        build_plot.plot_H_M(
            self.altitude,
            self.M_min_P,
            self.M_max_P,
            self.M_min_dop,
            self.M_Vi_max,
            self.M_OGR,
            save=run_save,
        )
        build_plot.plot_q_ch_q_km(
            self.altitude, self.q_km_min, self.q_ch_min, save=run_save
        )

    def climb_part(self, save_plot=False):
        M_0 = 1.2 * self.M_min_dop[0]
        const.Hk = 11
        self.H_nab = frmls.find_H_nab(self.altitude, const.Hk, self.Vy_max)
        Mk = frmls.find_Vk(self.H_nab, self.M_4)
        self.M_nab = frmls.find_M_nab(M_0, self.M_2, self.H_nab, Mk)
        a_H = self.df.get_column("a_H", "H", np.array([self.H_nab]), inter_value=True)
        self.V_nab = self.M_nab * a_H

        dVdH = frmls.dVdH_equation(self.V_nab, self.H_nab * 1000)
        n_x_nab = np.array([])
        self.Vy_max = np.array([])

        for i, alt in enumerate(self.H_nab):
            self.first_part(int(alt), save_plot=False, save_data=False)
            n_x_nab = np.append(
                n_x_nab, self.find_value_nab_des(self.n_x, self.V_nab[i])
            )

        GP_index = dh.get_min_or_max(self.K_n, min_or_max="max")
        self.K_gp = self.K_n[GP_index]
        self.V_gp = self.V[GP_index]
        self.Ce_gp = self.Ce[GP_index]
        self.Cy_gp = self.C_y_n[GP_index]

        k_nab = frmls.k_equation(self.V_nab, dVdH, const.G)
        self.teta_nab = frmls.teta_nab_equation(n_x_nab, k_nab)
        self.v_y_nab = frmls.v_y_nab_equation(self.Vy_max, k_nab)

        H_e = frmls.H_energy_equation(self.H_nab * 1000, self.V_nab, const.G)
        delta_H_e = frmls.delta_H_energy_equation(H_e)

        n_x_e = np.array([])
        self.Vy_max = np.array([])
        P_nab = np.array([])
        Ce_nab = np.array([])
        for i, alt in enumerate(self.H_nab):
            self.first_part(float(alt), save_plot=False, save_data=False)
            n_x_e = np.append(n_x_e, self.find_value_nab_des(self.n_x, self.V_nab[i]))
            P_nab = np.append(
                P_nab, self.find_value_nab_des(self.P_rasp, self.V_nab[i])
            )
            Ce_nab = np.append(Ce_nab, self.find_value_nab_des(self.Ce, self.V_nab[i]))

        n_x_avg = frmls.n_x_avg_equation(n_x_e)
        v_y_avg = frmls.v_y_avg_equation(self.Vy_max)
        CeP_Vy_avg = frmls.CeP_Vy_avg_equation(Ce_nab, P_nab, self.Vy_max)

        self.L_nab = frmls.L_nab_equation(delta_H_e, n_x_avg)
        self.t_nab = frmls.t_nab_equation(delta_H_e, v_y_avg)
        self.m_t_nab = frmls.m_t_equation(delta_H_e, CeP_Vy_avg)

        if save_plot:
            self.run_plot_climb_part()
        self.save_data_descent_climb_part(
            self.H_nab,
            self.M_nab,
            self.V_nab,
            dVdH,
            self.Vy_max,
            self.teta_nab,
            self.v_y_nab,
            H_e,
            delta_H_e,
            n_x_avg,
            v_y_avg,
            CeP_Vy_avg,
            n_x_nab,
            P_nab,
            Ce_nab,
            self.m_t_nab,
            self.L_nab,
            self.t_nab,
            move="climb",
        )

    def find_value_nab_des(self, value, V_nab):
        V_cl, value_cl = dh.remove_first_element_in_array(self.V, value)
        V_int, value_int = dh.interp_arrays(V_cl, value_cl)
        return value_int[dh.get_index_nearest_element_in_array(V_int, V_nab)]

    def run_plot_climb_part(self):
        plot_part3 = pbud(
            self.altitude, const.MACH, const.TYPE_NAMES, const.PATH_TO_DIRECTORY
        )
        plot_part3.plot_climb_param(
            self.t_nab, self.H_nab, self.teta_nab, self.v_y_nab, self.V_nab
        )
        plot_part3.plot_L_m(self.t_nab, self.L_nab, self.m_t_nab, file_name="L_m_climb")
        plot_part3.plot_H_M_profile(self.M_nab, self.H_nab, file_name="H_climb")

    def level_flight_part(self, debug=False):
        self.otn_m_t_nab = frmls.otn_m_t_nab_equation(sum(self.m_t_nab), const.M0)
        self.otn_m_t_kr = frmls.otn_m_t_kr_equation(
            const.OTN_M_CH,
            const.OTN_M_TSN,
            self.otn_m_t_nab,
            const.OTN_M_T_SNP,
            const.OTN_M_T_ANZ,
            const.OTN_M_T_PR,
        )
        T_kr = frmls.T_kr_equation(
            self.K_gp,
            const.G,
            self.Ce_gp,
            self.otn_m_t_nab,
            const.OTN_M_T_PR,
            self.otn_m_t_kr,
        )
        self.L_kr = frmls.L_kr_equation(
            self.V_gp,
            self.K_gp,
            const.G,
            self.Ce_gp,
            self.otn_m_t_nab,
            const.OTN_M_T_PR,
            self.otn_m_t_kr,
        )
        self.m_kkr = frmls.otn_m_kkr_equation(
            self.otn_m_t_nab, const.OTN_M_T_PR, self.otn_m_t_kr
        )
        Ro_H_kr = frmls.Ro_kr_equation(self.m_kkr, const.PS, self.Cy_gp, self.V_gp)
        self.H_k_kr = self.df.get_column(
            "H", "Ro_H", np.array([Ro_H_kr]), inter_value=True
        )

        dh.save_data(
            [T_kr, self.L_kr, Ro_H_kr, const.Hk, self.H_k_kr],
            text_handler.get_kr_table(),
            "level_flight_data.csv",
            const.PATH_TO_RESULTS,
        )
        dh.save_data_tex(
            [
                str(round(T_kr, 2)),
                str(round(self.L_kr, 0)),
                str(round(Ro_H_kr, 4)),
                str(round(const.Hk, 1)),
                str(round(self.H_k_kr[0], 1)),
            ],
            text_handler.get_kr_table_latex(),
            "level_flight_data.tex",
            const.PATH_TO_RESULTS,
            units_value=text_handler.get_kr_units_table(),
        )
        if debug:
            debug_level_flight_part(self.Ce_gp, T_kr, self.L_kr, Ro_H_kr, self.H_k_kr)

    def descent_part(self, save_plot=True):
        M_0 = 0.6
        const.Hk = 11
        self.H_des = np.flip(frmls.find_H_nab(const.H, const.Hk, self.Vy_max))

        self.M_des = frmls.find_M_des(M_0, self.M_1, self.H_des)

        a_H = self.df.get_column("a_H", "H", np.array([self.H_des]), inter_value=True)
        self.V_des = self.M_des * a_H
        dVdH = frmls.dVdH_equation(self.V_des, self.H_des * 1000)

        n_x_des = np.array([])
        self.Vy_max = np.array([])
        Vy_des = np.array([])

        for i, alt in enumerate(self.H_des):
            self.first_part(int(alt), save_plot=False, save_data=False)
            n_x_des = np.append(
                n_x_des, self.find_value_nab_des(self.n_x_des, self.V_des[i])
            )
            index_des = int(
                dh.get_index_nearest_element_in_array(const.MACH, self.M_des[i])
            )
            Vy_des = np.append(Vy_des, -self.V_y[index_des - 1])

        k_des = frmls.k_equation(self.V_des, dVdH, const.G)
        self.teta_des = frmls.teta_nab_equation(n_x_des, k_des)
        self.v_y_des = frmls.v_y_nab_equation(Vy_des, k_des)
        H_e = frmls.H_energy_equation(self.H_des * 1000, self.V_des, const.G)
        delta_H_e = frmls.delta_H_energy_equation(H_e)

        n_x_e = np.array([])
        self.Vy_max = np.array([])
        P_des = np.array([])
        Ce_des = np.array([])

        for i, alt in enumerate(H_e / 1000):
            self.first_part(int(alt), save_plot=False, save_data=False)
            n_x_e = np.append(
                n_x_e, self.find_value_nab_des(self.n_x_des, self.V_des[i])
            )
            P_des = np.append(P_des, self.find_value_nab_des(self.P_des, self.V_des[i]))
            Ce_des = np.append(
                Ce_des, self.find_value_nab_des(self.Ce_des, self.V_des[i])
            )
        n_x_avg = frmls.n_x_avg_equation(n_x_e)
        v_y_avg = frmls.v_y_avg_equation(Vy_des)
        CeP_Vy_avg = frmls.CeP_Vy_avg_equation(Ce_des, P_des, Vy_des)

        self.L_des = frmls.L_nab_equation(delta_H_e, n_x_avg)
        self.t_des = frmls.t_nab_equation(delta_H_e, v_y_avg)
        self.m_t_des = frmls.m_t_equation(delta_H_e, CeP_Vy_avg)

        self.H_plot, self.L_plot = self.prepare_data_for_H_L_tab()
        if save_plot:
            self.run_plot_descent_part()

        self.save_data_descent_climb_part(
            self.H_des,
            self.M_des,
            self.V_des,
            dVdH,
            self.Vy_max,
            self.teta_des,
            self.v_y_des,
            H_e,
            delta_H_e,
            n_x_avg,
            v_y_avg,
            CeP_Vy_avg,
            n_x_des,
            P_des,
            Ce_des,
            self.m_t_des,
            self.L_des,
            self.t_des,
            move="descent",
        )

    def save_data_descent_climb_part(
        self,
        H_move,
        M_move,
        V_move,
        dVdH,
        V_y_star,
        theta_move,
        V_y_move,
        H_e,
        delta_H_e,
        n_x_avg,
        v_y_avg,
        CeP_Vy_avg,
        n_x_move,
        P_move,
        Ce_move,
        m_t_move,
        L_move,
        t_move,
        move="descent",
    ):
        if move == "descent":
            file_name_mini_table = "descent_mini_table"
            file_name = "descent_data"
        elif move == "climb":
            file_name_mini_table = "climb_mini_table"
            file_name = "climb_data"
        else:
            raise ValueError(
                f'Can use only "descent" or "climb" value in move parameter, but given {move}'
            )

        dh.save_data(
            [sum(m_t_move), sum(L_move), sum(t_move)],
            text_handler.get_mini_table_3(descent_or_climb=move),
            file_name_mini_table + ".csv",
            const.PATH_TO_RESULTS,
        )
        dh.save_data_tex(
            [
                f"{round(sum(m_t_move), 1)}",
                f"{round(sum(L_move), 1)}",
                f"{round(sum(t_move),1)}",
            ],
            text_handler.get_mini_table_3_latex(descent_or_climb=move),
            file_name_mini_table + ".tex",
            const.PATH_TO_RESULTS,
            units_value=text_handler.get_mini_table_3_units(),
        )
        data_table = [
            [f"{round(val,1)}" for val in H_move],
            [f"{round(val,2)}" for val in M_move],
            [f"{round(val,1)}" for val in V_move],
            [f"{round(val,1)}" for val in V_move * 3.6],
            [f"{round(val,3)}" for val in dVdH],
            [f"{round(val,3)}" for val in n_x_move],
            [f"{round(val,1)}" for val in V_y_star],
            [f"{round(val,1)}" for val in theta_move],
            [f"{round(val,1)}" for val in V_y_move],
            [f"{round(val,0)}" for val in H_e],
            [f"{round(val,0)}" for val in delta_H_e],
            [f"{round(val,3)}" for val in 1 / n_x_avg],
            [f"{round(val,2)}" for val in delta_H_e / (1000 * n_x_move)],
            [f"{round(val,0)}" for val in P_move],  # part 2 table begining
            [f"{round(val,1)}" for val in (Ce_move * P_move) / V_y_move],
            [f"{round(val,1)}" for val in CeP_Vy_avg],
            [f"{round(val,1)}" for val in m_t_move],
            [f"{round(val,1)}" for val in L_move],
            [f"{round(val,1)}" for val in v_y_avg],
            [f"{round(val,2)}" for val in t_move],
            [f"{round(val,3)}" for val in Ce_move],
        ]
        dh.save_data(
            data_table,
            text_handler.get_row_name_table_3(descent_or_climb=move),
            file_name + ".csv",
            const.PATH_TO_RESULTS,
        )
        units_part1 = np.array(text_handler.get_row_units_table_3_latex()[0])
        units_part2 = np.array(text_handler.get_row_units_table_3_latex()[1])

        dh.save_data_tex(
            data_table[0:13],
            text_handler.get_row_name_table_3_tex(descent_or_climb=move)[0],
            file_name + ".tex",
            const.PATH_TO_RESULTS,
            units_value=units_part1,
        )
        dh.save_data_tex(
            data_table[13:],
            text_handler.get_row_name_table_3_tex(descent_or_climb=move)[1],
            file_name + "_part2.tex",
            const.PATH_TO_RESULTS,
            units_value=units_part2,
        )

    def prepare_data_for_H_L_tab(self):
        L_nab_plus_kr = np.append(self.L_nab[:-1], self.L_kr)
        L_plus_des = np.append(L_nab_plus_kr, self.L_des)
        H_nab_plus_kr = np.append(self.H_nab, self.H_k_kr)
        H_plus_des = np.append(H_nab_plus_kr, self.H_des[1:])
        return H_plus_des, dh.sum_array(L_plus_des)

    def run_plot_descent_part(self):
        run_plot = pbud(
            self.altitude, const.MACH, const.TYPE_NAMES, const.PATH_TO_DIRECTORY
        )
        run_plot.plot_climb_param(
            self.t_des,
            self.H_des,
            self.teta_des,
            self.v_y_des,
            self.V_des,
            file_name="descent_params",
        )
        run_plot.plot_L_m(self.t_des, self.L_des, self.m_t_des, "L_m_des")
        run_plot.plot_H_M_profile(self.M_des, self.H_des, "H_des")
        run_plot.plot_L_H(
            self.H_plot,
            self.L_plot,
        )

    def cargo_possibility(self, save_plot=False):
        m_tsn_mode1 = const.OTN_M_CH * const.M0
        otn_m_kr_mode2 = frmls.otn_m_t_kr_mode_equation(
            const.OTN_M_T,
            self.otn_m_t_nab,
            sum(self.m_t_des) / const.M0,
            const.OTN_M_T_ANZ,
            const.OTN_M_T_PR,
        )
        L_des_nab = np.sum(np.append(self.L_des, self.L_nab))
        L_kr_mode2 = frmls.L_kr_equation(
            self.V_gp,
            self.K_gp,
            const.G,
            self.Ce_gp,
            self.otn_m_t_nab,
            const.OTN_M_T_PR,
            otn_m_kr_mode2,
        )
        m_tsn_mode2 = abs((1 - const.OTN_M_EMPTY - const.OTN_M_T) * const.M0)
        otn_m_vz = const.OTN_M_EMPTY + const.OTN_M_T
        L_kr_mode3 = frmls.L_kr_equation(
            self.V_gp,
            self.K_gp,
            const.G,
            self.Ce_gp,
            self.otn_m_t_nab,
            const.OTN_M_T_PR,
            otn_m_kr_mode2,
            otn_m_vz=otn_m_vz,
        )
        L_array = np.array(
            [
                np.max(self.L_plot),
                np.sum(np.append(L_kr_mode2, L_des_nab)),
                np.sum(np.append(L_kr_mode3, L_des_nab)),
            ]
        )
        m_cargo_array = np.array([m_tsn_mode1, m_tsn_mode2, 0])
        configuration_numbers = np.arange(1, len(L_array) + 1)
        dh.save_data(
            [L_array, m_cargo_array],
            text_handler.get_row_cargo_table(),
            "cargo_load.csv",
            const.PATH_TO_RESULTS,
        )
        dh.save_data_tex(
            [
                configuration_numbers,
                [f"{round(val,0)}" for val in L_array],
                [f"{round(val,0)}" for val in m_cargo_array],
            ],
            text_handler.get_row_cargo_table_latex(),
            "cargo_load.tex",
            const.PATH_TO_RESULTS,
            units_value=text_handler.get_row_units_cargo_table(),
        )
        if save_plot:
            self.run_plot_cargo_part(L_array, m_cargo_array)

    def run_plot_cargo_part(self, L, m):
        plot_cargo = pbud(
            self.altitude, const.MACH, const.TYPE_NAMES, const.PATH_TO_DIRECTORY
        )
        plot_cargo.plot_cargo(L, m)

    def take_off_landing_part(self):
        self.first_part(0)
        P_takeoff = 1.25 * self.P_rasp
        otn_P_takeoff = P_takeoff[0] / (const.M0 * const.G)
        Cy_rotate = self.df.get_column(
            "Cy_a_type1", "alpha_type1", np.array([const.ALPHA_RT]), inter_value=True
        )
        Cx_rotate = self.df.get_column(
            "Cx_type1", "Cy_type1", np.array([Cy_rotate]), inter_value=True
        )
        Cp = frmls.Cp_equation(otn_P_takeoff, const.F_TO)
        Cy_run = self.df.get_column(
            "Cy_a_type1", "alpha_type1", np.array([const.ALPHA_TO]), inter_value=True
        )
        Cx_run = self.df.get_column(
            "Cx_type1", "Cy_type1", np.array([Cy_run]), inter_value=True
        )
        bp = frmls.bp_equation(Cx_run, const.F_TO, Cy_run, self.Ro_H, const.PS)
        V_rotate = frmls.V_rotate_equation(
            const.PS, otn_P_takeoff, const.ALPHA_RT, self.Ro_H, Cy_rotate
        )
        L_run = frmls.L_takeoff_equation(const.G, bp, Cp, V_rotate)
        V2 = frmls.V2_equation(V_rotate)
        V_avg = frmls.hat_V_avg_equation(V2, V_rotate)
        nx_avg = frmls.hat_nx_avg_equation(
            otn_P_takeoff, Cx_rotate, self.Ro_H, V_avg, const.PS
        )
        L_vuv = frmls.L_vuv_equation(nx_avg, V2, V_rotate, const.G, const.H_TO)
        otn_m_landing = frmls.otn_m_landing_equation(self.m_kkr, const.OTN_M_T_SNP)
        Cy_touch = self.df.get_column(
            "Cy_a_type2", "alpha_type2", np.array([const.ALPHA_TD]), inter_value=True
        )
        V_touch = frmls.v_touch_equation(otn_m_landing, const.PS, Cy_touch, self.Ro_H)
        P_rev = frmls.P_rev_equation(
            const.OTN_P_0, const.N_REV, const.N_DV, otn_m_landing * const.M0
        )
        otn_P_rev = frmls.otn_P_rev_equation(P_rev, otn_m_landing * const.M0, const.G)
        a_n = frmls.a_n_equation(otn_P_rev, const.F_LA)
        Cy_ground = self.df.get_column(
            "Cy_a_type3", "alpha_type3", np.array([const.ALPHA_LA]), inter_value=True
        )
        Cx_ground = self.df.get_column(
            "Cx_type3", "Cy_type3", np.array([Cy_ground]), inter_value=True
        )

        b_n = frmls.b_n_equation(
            self.Ro_H, otn_m_landing, const.PS, Cx_ground, const.F_LA, Cy_ground
        )
        L_ground = frmls.L_run_equation(const.G, b_n, a_n, V_touch)
        Cy_landing = frmls.Cy_landing_equation(Cy_touch)
        Cx_landing = self.df.get_column(
            "Cx_type2", "Cy_type2", np.array([Cy_landing]), inter_value=True
        )

        V_flare = frmls.V_flare_equation(otn_m_landing, const.PS, Cy_landing, self.Ro_H)
        K_landing = frmls.K_landing_equation(Cy_landing, Cx_landing)
        L_vup = frmls.L_vup_equation(K_landing, const.H_LA, V_flare, V_touch, const.G)
        L_landing = frmls.L_landing_equation(L_ground, L_vup)
        self.save_takeoff_landing_data(
            [
                f"{round(V_rotate, 0)}",
                f"{round(L_run[0],0)}",
                f"{round(L_run[0] + L_vuv[0],0)}",
                f"{round(V_touch,0)}",
                f"{round(L_ground[0],0)}",
                f"{round(L_landing[0],0)}",
            ]
        )

    def save_takeoff_landing_data(self, data):
        dh.save_data(
            data,
            text_handler.get_to_la_table(),
            "takeoff_landing_table.csv",
            const.PATH_TO_RESULTS,
        )
        dh.save_data_tex(
            data,
            text_handler.get_to_la_table_latex(),
            "takeoff_landing_table.tex",
            const.PATH_TO_RESULTS,
            units_value=text_handler.get_to_la_units_table(),
        )

    def turn_part(self, save_plot=False):
        self.first_part(const.TURN_H)

        MACH = frmls.turn_mach(const.TURN_M_MIN, const.TURN_M_MAX, self.M_flying)
        MACH = MACH.astype(np.float32)

        index_MACH_out = np.array(
            [
                i
                for i, val in enumerate(
                    np.isin(MACH, const.MACH_output, assume_unique=True)
                )
                if val == True
            ]
        )
        index_MACH_out = np.append(index_MACH_out, len(MACH))

        TURN_Cy_dop = self.df.get_column("Cydop", "M", MACH, inter_value=True)
        TURN_Ro = self.df.get_column(
            "Ro_H", "H", np.array([const.TURN_H]), inter_value=True
        )
        TURN_a_sos = self.df.get_column(
            "a_H", "H", np.array([const.TURN_H]), inter_value=True
        )
        TURN_A = self.df.get_column("A", "M", MACH, inter_value=True)
        TURN_Cym = self.df.get_column("Cym", "M", MACH, inter_value=True)
        TURN_Cxm = self.df.get_column("Cxm", "M", MACH, inter_value=True)

        V = frmls.v_speed(MACH, TURN_a_sos)

        q = frmls.q_dynamic_pressure(V, TURN_Ro)
        otn_m_plane = frmls.otn_m_plane_equation(const.OTN_M_T)
        TURN_Cy_GP = frmls.C_y_n_lift_coefficient(otn_m_plane, const.PS, q)
        TURN_Cx_GP = frmls.C_x_n_drag_coefficient(
            TURN_Cxm, TURN_A, TURN_Cy_GP, TURN_Cym
        )
        TURN_K_GP = frmls.K_n_lift_to_drag_ratio(TURN_Cy_GP, TURN_Cx_GP)

        n_y = frmls.n_y_equation(TURN_Cy_dop, TURN_Cy_GP)
        n_y_dop = dh.find_min_max_from_arrays(n_y, const.TURN_n_ye)
        TURN_MASS = otn_m_plane * const.M0

        TURN_P_rasp = frmls.P_rasp_equation(
            const.OTN_P_0,
            TURN_MASS,
            const.G,
            self.find_tilda_P(MACH, const.TURN_H),
            const.TURN_H,
            self.p_h_11,
            TURN_Ro,
        )
        TURN_P_potr = frmls.P_potr_equation(otn_m_plane, TURN_MASS, const.G, TURN_K_GP)
        otn_P = frmls.otn_P_equation(TURN_P_rasp, TURN_MASS, const.G)
        n_y_p = frmls.n_y_p_equation(TURN_Cy_GP, TURN_Cym, otn_P, TURN_Cxm, TURN_A)
        n_y_turn = dh.find_min_max_from_arrays(n_y_p, n_y_dop)

        omega_turn = frmls.turn_omega_equation(const.G, V, n_y_turn)
        r_turn = frmls.turn_radius_equation(V, omega_turn)
        t_turn = frmls.turn_time_equation(r_turn, V)

        self.save_turn_data(
            np.array(
                [
                    [
                        f"{np.format_float_positional(val,2)}"
                        for i, val in enumerate(MACH)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,0)}"
                        for i, val in enumerate(V)
                        if i in index_MACH_out
                    ],
                    [
                        f"{np.format_float_positional(val*3.6,0)}"
                        for i, val in enumerate(V)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,0)}"
                        for i, val in enumerate(q)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,3)}"
                        for i, val in enumerate(TURN_Cy_GP)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,3)}"
                        for i, val in enumerate(TURN_Cy_dop)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,3)}"
                        for i, val in enumerate(n_y_dop)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,2)}"
                        for i, val in enumerate(TURN_K_GP)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val/10000,3)}"
                        for i, val in enumerate(TURN_P_potr)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val/10000,3)}"
                        for i, val in enumerate(TURN_P_rasp)
                        if i in index_MACH_out
                    ],
                ]
            ),
            np.array(
                [
                    [
                        f"{round(val,3)}"
                        for i, val in enumerate(otn_P)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,3)}"
                        for i, val in enumerate(n_y_p)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,3)}"
                        for i, val in enumerate(n_y_turn)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,3)}"
                        for i, val in enumerate(omega_turn)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,1)}"
                        for i, val in enumerate(r_turn)
                        if i in index_MACH_out
                    ],
                    [
                        f"{round(val,1)}"
                        for i, val in enumerate(t_turn)
                        if i in index_MACH_out
                    ],
                ]
            ),
        )
        if save_plot:
            self.run_plot_turn_part(MACH, n_y_turn, omega_turn, r_turn, t_turn)

    def save_turn_data(self, data1, data2):
        dh.save_data(
            data1,
            text_handler.get_row_name_turn_table_part_1(),
            "turn_data_table_part1.csv",
            const.PATH_TO_RESULTS,
        )
        dh.save_data_tex(
            data1,
            text_handler.get_row_name_turn_table_part_1(latex=True),
            "turn_data_table_part1.tex",
            const.PATH_TO_RESULTS,
            units_value=text_handler.get_row_name_turn_table_part_1_units(),
        )
        dh.save_data(
            data2,
            text_handler.get_row_name_turn_table_part_2(),
            "turn_data_table_part2.csv",
            const.PATH_TO_RESULTS,
        )
        dh.save_data_tex(
            data2,
            text_handler.get_row_name_turn_table_part_2(latex=True),
            "turn_data_table_part2.tex",
            const.PATH_TO_RESULTS,
            units_value=text_handler.get_row_name_turn_table_part_2_units(),
        )

    def run_plot_turn_part(self, M, ny, omega, r, t):
        run_plot = pbud(self.altitude, M, const.TYPE_NAMES, const.PATH_TO_DIRECTORY)
        run_plot.plot_turn(ny, omega, r, t)

    def static_stability_control_part(self, H_pr, H_st, save_plot=False):
        SSC_M = 0.2
        self.__define_variables()
        self.first_part(0)
        self.second_part(0, H_pr, H_st)

        mach_values = dh.proper_array(self.M_min[0], self.M_max[0], 0.01)
        otn_x_T = self.find_otn_x_T(SSC_M, save_plot=save_plot)

        otn_S_go = self.otn_S_go_star

        otn_x_F = np.array([])
        otn_x_H = np.array([])
        otn_x_TPZ = np.array([])
        sigma_n = np.array([])

        for M in mach_values:
            self.find_otn_x_T(M, otn_S_go)
            otn_x_F = np.append(otn_x_F, self.otn_x_f)
            otn_x_H = np.append(otn_x_H, self.otn_x_H)
            otn_x_TPZ = np.append(otn_x_TPZ, self.otn_x_TPZ)
            sigma_n = np.append(
                sigma_n,
                frmls.sigma_n_equation(otn_x_T, self.otn_x_f, self.m_z_w_z, self.mu),
            )

        out_index = [
            dh.get_index_nearest_element_in_array(mach_values, value)
            for value in const.MACH_output
        ]
        out_index = np.unique([0] + out_index + [len(mach_values)])

        if save_plot:
            run_plot = pbud(0, mach_values, const.TYPE_NAMES, const.PATH_TO_DIRECTORY)
            run_plot.plot_xs_sigmas(otn_x_F, otn_x_H, otn_x_TPZ, sigma_n, mach_values)

        dh.save_data(
            np.array(
                [
                    mach_values,
                    otn_x_F,
                    otn_x_H,
                    otn_x_TPZ,
                    sigma_n,
                ]
            ),
            text_handler.get_row_name_sigmas(),
            "sigmas_table.csv",
            const.PATH_TO_RESULTS,
        )
        dh.save_data_tex(
            np.array(
                [
                    [
                        f"{np.format_float_positional(i, precision=2)}"
                        for index, i in enumerate(mach_values)
                        if index in out_index
                    ],
                    [
                        f"{round(i,4)}"
                        for index, i in enumerate(otn_x_F)
                        if index in out_index
                    ],
                    [
                        f"{round(i,4)}"
                        for index, i in enumerate(otn_x_H)
                        if index in out_index
                    ],
                    [
                        f"{round(i,4)}"
                        for index, i in enumerate(otn_x_TPZ)
                        if index in out_index
                    ],
                    [
                        f"{round(i,4)}"
                        for index, i in enumerate(sigma_n)
                        if index in out_index
                    ],
                ]
            ),
            text_handler.get_row_name_sigmas_latex(),
            "sigmas_table.tex",
            const.PATH_TO_RESULTS,
        )

        alts = np.array([0, 6, const.Hk])
        mach_speeds = []
        fi_bal_array = []
        fi_n_array = []
        ny_p_array = []

        for alt in alts:
            self.__define_variables()
            self.first_part(alt)
            self.second_part(alt, H_pr, H_st)
            mach_values = dh.proper_array(self.M_min[0], self.M_max[0], 0.01)
            mach_speeds.append(mach_values)
            fi_bal, fi_n, ny_p = self.find_phis(alt, otn_S_go, otn_x_T, mach_values)
            fi_bal_array.append(fi_bal)
            fi_n_array.append(fi_n)
            ny_p_array.append(ny_p)

        ny_dops = self.find_ny_dops(alts, mach_speeds)
        print(ny_dops)
        self.run_plot_phis_part(
            alts, mach_speeds, fi_bal_array, fi_n_array, ny_p_array, ny_dops
        )

    def find_ny_dops(self, alts, mach):
        ny_dops = []
        for i, alt in enumerate(alts):
            a_sos = self.df.get_column("a_H", "H", np.array([alt]), inter_value=True)
            Ro = self.df.get_column("Ro_H", "H", np.array([alt]), inter_value=True)
            V = frmls.v_speed(mach[i], a_sos)
            q = frmls.q_dynamic_pressure(V, Ro)
            otn_m_plane = frmls.otn_m_plane_equation(const.OTN_M_T)
            Cy_GP = frmls.C_y_n_lift_coefficient(otn_m_plane, const.PS, q)
            Cy_dop = self.df.get_column("Cydop", "M", mach[i], inter_value=True)
            n_y = frmls.n_y_equation(Cy_dop, Cy_GP)
            n_y_dop = dh.find_min_max_from_arrays(n_y, const.TURN_n_ye)
            ny_dops.append(n_y_dop)
        return ny_dops

    def find_phis(self, alt, otn_S_go, otn_x_T, mach_values):
        self.take_constant_stability(mach_values)
        V_value = self.a_H[0] * mach_values
        self.find_otn_x_T(mach_values, otn_S_go)

        m_z_Cy = frmls.m_z_Cy_equation(otn_x_T, self.otn_x_f)
        m_z_delta = frmls.m_z_delta_equation(
            self.Cy_go_a_go, otn_S_go, const.OTN_L_GO, self.SSC_K_go, self.n_v
        )

        otn_m_plane = frmls.otn_m_plane_equation(const.OTN_M_T)
        q_dynamic = frmls.q_dynamic_pressure(V_value, self.Ro_H)

        Cy_gp = frmls.C_y_n_lift_coefficient(otn_m_plane, const.PS, q_dynamic)
        mz0 = frmls.m_z_0_equation(
            self.mz0_bgo,
            otn_S_go,
            const.OTN_L_GO,
            self.SSC_K_go,
            self.Cy_go_a_go,
            self.a0,
            self.epsilon_a,
        )

        fi_bal = frmls.phi_bal_equation(
            mz0, m_z_Cy, Cy_gp, m_z_delta, const.OTN_L_GO, const.FI_UST, self.n_v
        )
        sigma_n = frmls.sigma_n_equation(otn_x_T, self.otn_x_f, self.m_z_w_z, self.mu)
        fi_n = frmls.phi_n_equation(Cy_gp, sigma_n, m_z_delta)
        nyp = frmls.nyp_equation(const.FI_MAX, const.FI_UST, fi_bal, fi_n)

        out_index = [
            dh.get_index_nearest_element_in_array(mach_values, value)
            for value in const.MACH_output
        ]
        out_index = np.unique([0] + out_index + [len(mach_values)])

        DATA = np.array(
            [
                [
                    f"{round(i,2)}"
                    for index, i in enumerate(mach_values)
                    if index in out_index
                ],
                [
                    f"{round(i,0)}"
                    for index, i in enumerate(V_value)
                    if index in out_index
                ],
                [
                    f"{round(i,2)}"
                    for index, i in enumerate(fi_bal)
                    if index in out_index
                ],
                [f"{round(i,2)}" for index, i in enumerate(fi_n) if index in out_index],
                [f"{round(i,3)}" for index, i in enumerate(nyp) if index in out_index],
            ]
        )
        self.save_data_phi(DATA, alt)
        return fi_bal, fi_n, nyp

    def save_data_phi(self, data, H):
        dh.save_data(
            data,
            text_handler.get_row_name_phis_table(),
            f"phi_table_H={H}.csv",
            const.PATH_TO_RESULTS,
        )
        dh.save_data_tex(
            data,
            text_handler.get_row_name_phis_table_latex(),
            f"phi_table_H={H}.tex",
            const.PATH_TO_RESULTS,
            units_value=text_handler.get_row_name_phis_table_units(),
        )

    def find_otn_x_T(self, ssc_M, otn_S_go=None, save_plot=False):
        self.take_constant_stability(ssc_M)
        if not otn_S_go:
            otn_S_go = const.OTN_S_GO
            mz0_bgo = const.MZ0_BGO_LA
        else:
            mz0_bgo = self.mz0_bgo

        delta_otn_x_f = frmls.delta_otn_x_f_equation(
            self.Cy_go_a_go,
            self.Cy_a,
            self.epsilon_a,
            otn_S_go,
            const.OTN_L_GO,
            self.SSC_K_go,
        )
        self.otn_x_f = frmls.otn_x_f_equation(
            self.otn_x_f_bgo,
            delta_otn_x_f,
        )
        m_z_go_w_z = frmls.m_otn_omega_z_z_go_equation(
            self.cy_a_go_go,
            otn_S_go,
            const.OTN_L_GO,
            self.SSC_K_go,
        )

        self.m_z_w_z = frmls.m_z_otn_omega_z_equation(self.m_z_bgo_w_z, m_z_go_w_z)
        self.mu = frmls.mu_equation(const.PS, self.Ro_H, const.G, const.B_A)
        self.otn_x_H = frmls.otn_x_n_equaiton(self.otn_x_f, self.m_z_w_z, self.mu)
        self.otn_x_TPZ = frmls.otn_x_tpz_equation(self.otn_x_H, const.SIGMA_N_MIN)

        if not isinstance(otn_S_go, collections.abc.Iterable):
            return True

        phi_ef = frmls.phi_ef_equation(const.FI_UST, self.n_v, const.FI_MAX)
        Cy_go = frmls.Cy_go_equation(
            self.cy_a_go_go, const.ALPHA_LA, self.epsilon_a, phi_ef
        )
        Cy_bgo = frmls.Cy_bgo_equation(self.Cy0_bgo, self.Cya_bgo, const.ALPHA_LA)
        otn_x_TPP = frmls.otn_x_tpp_equation(
            mz0_bgo,
            self.otn_x_f_bgo,
            Cy_bgo,
            Cy_go,
            otn_S_go,
            self.SSC_K_go,
            const.OTN_L_GO,
        )

        self.otn_S_go_star = dh.find_diff_in_two_plot_by_target(
            self.otn_x_TPZ, otn_x_TPP, otn_S_go, const.DELTA_OTN_X_E * 1.2
        )

        fun_otn_x_TPP = dh.find_linear_func(otn_S_go, otn_x_TPP)
        fun_otn_x_TPZ = dh.find_linear_func(otn_S_go, self.otn_x_TPZ)
        xtpz_star = fun_otn_x_TPZ(self.otn_S_go_star)
        xtpp_star = fun_otn_x_TPP(self.otn_S_go_star)

        if save_plot:
            self.run_plot_stability_control_part(
                otn_S_go,
                otn_x_TPP,
                self.otn_x_TPZ,
                xtpz_star,
                xtpp_star,
                self.otn_S_go_star,
            )
            dh.save_data(
                np.array([otn_S_go, otn_x_TPP, self.otn_x_TPZ]),
                text_handler.get_row_name_otn_S_go(),
                f"otn_S_go.csv",
                const.PATH_TO_RESULTS,
            )
            dh.save_data_tex(
                np.array(
                    [
                        [f"{round(i,2)}" for i in otn_S_go],
                        [f"{round(i,4)}" for i in otn_x_TPP],
                        [f"{round(i,4)}" for i in self.otn_x_TPZ],
                    ]
                ),
                text_handler.get_row_name_otn_S_go_latex(),
                f"otn_S_go.tex",
                const.PATH_TO_RESULTS,
            )

        return frmls.otn_x_t_equation(xtpz_star, xtpp_star)

    def take_constant_stability(self, mach):
        self.SSC_K_go = self.df.get_column(
            "K_go", "M_stab", np.array([mach]), inter_value=True
        )
        self.Cy_go_a_go = self.df.get_column(
            "C_alphago_y_go", "M_stab", np.array([mach]), inter_value=True
        )
        self.epsilon_a = self.df.get_column(
            "e_alpha", "M_stab", np.array([mach]), inter_value=True
        )
        self.Cy_a = self.df.get_column("Cya", "M", np.array([mach]), inter_value=True)
        self.otn_x_f_bgo = self.df.get_column(
            "otn_x_f_bgo", "M_stab", np.array([mach]), inter_value=True
        )
        self.cy_a_go_go = self.df.get_column(
            "C_alphago_y_go", "M_stab", np.array([mach]), inter_value=True
        )
        self.m_z_bgo_w_z = self.df.get_column(
            "mz_otn_omega_z_bgo", "M_stab", np.array([mach]), inter_value=True
        )
        self.n_v = self.df.get_column(
            "n_v", "M_stab", np.array([mach]), inter_value=True
        )
        self.Cy0_bgo = self.df.get_column(
            "Cy0_bgo", "M_stab", np.array([mach]), inter_value=True
        )
        self.Cya_bgo = self.df.get_column(
            "Cya_bgo", "M_stab", np.array([mach]), inter_value=True
        )
        self.mz0_bgo = self.df.get_column(
            "mz0_bgo", "M_stab", np.array([mach]), inter_value=True
        )
        self.a0 = self.df.get_column("a0", "M", np.array([mach]), inter_value=True)

    def run_plot_stability_control_part(
        self, otn_S_go, otn_xtpp, otn_xtpz, xtpz_star, xtpp_star, S_star
    ):
        run_plot = pbud(
            self.altitude, const.MACH, const.TYPE_NAMES, const.PATH_TO_DIRECTORY
        )
        run_plot.plot_center_value(
            otn_S_go, otn_xtpp, otn_xtpz, xtpz_star, xtpp_star, S_star
        )

    def run_plot_phis_part(self, alts, mach, fi_bal, fi_n, ny_p, ny_dops):
        run_plot = pbud(
            self.altitude, const.MACH, const.TYPE_NAMES, const.PATH_TO_DIRECTORY
        )
        run_plot.plot_phi_bal(alts, mach, fi_bal)
        run_plot.plot_phi_n(alts, mach, fi_n)
        run_plot.plot_ny_p(alts, mach, ny_p, ny_dops)

    def __define_variables(self):
        self.M_min_P = np.array([])
        self.M_max_P = np.array([])
        self.M_min_dop = np.array([])
        self.Vy_max = np.array([])
        self.q_km_min = np.array([])
        self.q_ch_min = np.array([])
        self.M_1 = np.array([])
        self.M_2 = np.array([])
        self.V_3 = np.array([])
        self.V_4 = np.array([])
        self.M_4 = np.array([])
        self.Vy_check = np.array([])


def debug_level_flight_part(Ce_gp, T_kr, L_kr, Ro, H):
    print(f"Удельной расход топлива:{Ce_gp}")
    print(f"Время крейсерского полета: {T_kr} мин")
    print(f"Дальность крейсерского полета: {L_kr} км")
    print(f"Densety: {Ro} :)")
    print(f"Высота в конце крейсерского полета: {H}")


def find_celling(calc):
    def value_find(value):
        step = 1
        altitude = const.H[0]
        iterations = np.array([])
        find = False
        while not find:
            alt = altitude + step
            Vy_check = calc.get_V_y(alt)
            Vy_check = dh.remove_first_element_in_array(Vy_check)
            iter1 = Vy_check[dh.get_min_or_max(Vy_check, min_or_max="max")]
            if iter1 > value:
                altitude = alt
                iterations = np.append(iterations, iter1)
            else:
                step = step / 2

            try:
                if abs(iterations[-1] - iterations[-2]) <= 0.00001:
                    find = True
            except Exception as e:
                find = False

        return altitude

    H_st = value_find(0)
    H_pr = value_find(0.5)
    return H_st, H_pr
