import numpy as np
import matplotlib.pyplot as plt

from DataHandler import DataHandler as dh
from .text_engine import TextHandler as text_handler
from .plot_engine import Plotter as plot


class PlotBuilderUsingData:
    def __init__(self, alt, M, type_names, save_path):
        self.altitude = alt
        self.pth = text_handler(self.altitude)
        self.MACH = M
        self.TYPE_NAMES = type_names
        self.save_path = save_path

    def plot_P(self, P_potr, P_rasp, text_side, save=True):
        MACH_int, P_potr_int, P_rasp_int = dh.prepare_data_for_plot(
            self.MACH, P_potr, P_rasp, remove_first_element=True
        )
        cross_position = dh.get_crossing_point(P_rasp_int, P_potr_int)

        min_position = dh.get_min_or_max(P_potr_int)
        M_1, Ppmin = MACH_int[min_position], P_potr_int[min_position]
        try:
            MminP, MmaxP = MACH_int[cross_position]
        except ValueError as e:
            result = MACH_int[cross_position]
            if result > M_1:
                MmaxP = result
                MminP = 0
            elif result < M_1:
                MminP = result
                MmaxP = 0
            else:
                MminP = 0
                MmaxP= 0

        for type_name in self.TYPE_NAMES:
            ploter_P_p_P_r = plot(
                MACH_int, save_type=type_name, fun1=P_rasp_int, fun2=P_potr_int
            )
            ploter_P_p_P_r.get_figure(
                self.pth.get_label_in_box("P")[0],
                self.pth.get_label_in_box("P")[1],
            )
            ploter_P_p_P_r.add_labels(
                self.pth.get_label("M"),
                self.pth.get_label("P"),
            )
            plt.ylim([0, P_rasp_int[cross_position][0] * 1.35])
            if len(cross_position) == 1:
                ploter_P_p_P_r.add_text(
                    MACH_int,
                    P_rasp_int,
                    cross_position,
                    self.pth.get_plot_text("P")[1],
                )
            else:
                ploter_P_p_P_r.add_text(
                    MACH_int,
                    P_rasp_int,
                    cross_position[0],
                    self.pth.get_plot_text("P")[0],
                    side=text_side,
                )
                ploter_P_p_P_r.add_text(
                    MACH_int,
                    P_rasp_int,
                    cross_position[1],
                    self.pth.get_plot_text("P")[1],
                )

            ploter_P_p_P_r.add_text(
                MACH_int,
                P_potr_int,
                min_position,
                self.pth.get_plot_text("P")[-1],
            )
            ploter_P_p_P_r.set_legend(loc_code=3)
            ploter_P_p_P_r.set_notation(4)
            if save:
                ploter_P_p_P_r.save_figure(
                    f"P_H={round(self.altitude, 4)}", self.save_path
                )
            ploter_P_p_P_r.close_plot()

        return MminP, MmaxP, M_1

    def plot_C_y_C_dop(self, C_y_n, Cy_dop, save=True):
        MACH_int, Cy_dop_int, C_y_n_int = dh.prepare_data_for_plot(
            self.MACH, Cy_dop, C_y_n, remove_first_element=True
        )

        cross_position = dh.get_crossing_point(Cy_dop_int, C_y_n_int)
        M_min_dop = MACH_int[cross_position]

        for type_name in self.TYPE_NAMES:
            plotter_C_y = plot(
                MACH_int, save_type=type_name, fun1=Cy_dop_int, fun2=C_y_n_int
            )
            plotter_C_y.get_figure(
                self.pth.get_label_in_box("C_y")[0],
                self.pth.get_label_in_box("C_y")[1],
            )
            plotter_C_y.add_labels(
                self.pth.get_label("M"),
                self.pth.get_label("C_y"),
            )
            plt.ylim([0, C_y_n_int[cross_position] + C_y_n_int[cross_position] * 0.15])
            plotter_C_y.add_text(
                MACH_int, C_y_n_int, cross_position, self.pth.get_plot_text("C_y")[0]
            )
            plotter_C_y.set_legend(loc_code=3)
            if save:
                plotter_C_y.save_figure(
                    f"Cy_H={round(self.altitude, 4)}", self.save_path
                )
            plotter_C_y.close_plot()
        return M_min_dop

    def plot_V_y(self, V_y, save=True):
        (
            MACH_int,
            V_y_int,
        ) = dh.prepare_data_for_plot(self.MACH, V_y, remove_first_element=True)
        max_position = dh.get_min_or_max(V_y_int, min_or_max="max")
        M_2, Vy_max = MACH_int[max_position], V_y_int[max_position]
        for type_name in self.TYPE_NAMES:
            plotter_Vy = plot(MACH_int, save_type=type_name, fun1=V_y_int)
            plotter_Vy.get_figure(
                self.pth.get_label_in_box("V_y")[0],
            )
            plotter_Vy.add_labels(
                self.pth.get_label("V"),
                self.pth.get_label("V_y"),
            )
            plt.ylim([0, Vy_max + Vy_max * 0.15])
            plotter_Vy.add_text(
                MACH_int,
                V_y_int,
                max_position,
                self.pth.get_plot_text("V_y")[0],
                add_value="y",
            )
            plotter_Vy.set_legend(loc_code=3)
            if save:
                plotter_Vy.save_figure(
                    f"V_y_H={round(self.altitude, 4)}", self.save_path
                )
            plotter_Vy.close_plot()
        return M_2, Vy_max

    def plot_q_ch(self, V, q_ch, save=True):
        if self.altitude >= 11:
            (
                V_int,
                q_ch_int,
            ) = dh.prepare_data_for_plot(V, q_ch, remove_first_element=True)
        else:
            (
                V_int,
                q_ch_int,
            ) = dh.prepare_data_for_plot(V, q_ch)
        min_position = dh.get_min_or_max(q_ch_int)
        V_3, q_ch_min = V_int[min_position], q_ch_int[min_position]
        for type_name in self.TYPE_NAMES:
            plotter_q_ch = plot(V_int, save_type=type_name, fun1=q_ch_int)
            plotter_q_ch.get_figure(self.pth.get_label_in_box("q_ch")[0])
            plt.ylim([q_ch_min - q_ch_min * 0.05, q_ch_int[-1] + q_ch_int[-1] * 0.2])
            plotter_q_ch.add_labels(
                self.pth.get_label("V"),
                self.pth.get_label("q_ch")[0],
            )
            plotter_q_ch.add_text(
                V_int,
                q_ch_int,
                min_position,
                self.pth.get_plot_text("q_ch")[0],
            )
            plotter_q_ch.set_legend()
            plotter_q_ch.set_notation(4)
            if save:
                plotter_q_ch.save_figure(
                    f"q_ch_H={round(self.altitude, 4)}", self.save_path
                )
            plotter_q_ch.close_plot()

        return V_3, q_ch_min

    def plot_q_km(self, V, q_km, save=True):
        if self.altitude >= 11:
            (
                V_int,
                q_km_int,
            ) = dh.prepare_data_for_plot(V, q_km, remove_first_element=True)
        else:
            (
                V_int,
                q_km_int,
            ) = dh.prepare_data_for_plot(V, q_km)

        min_position = dh.get_min_or_max(q_km_int)
        V_4, q_km_min = V_int[min_position], q_km_int[min_position]
        for type_name in self.TYPE_NAMES:
            plotter_q_km = plot(V_int, save_type=type_name, fun1=q_km_int)
            plotter_q_km.get_figure(self.pth.get_label_in_box("q_km")[0])
            plt.ylim([q_km_min - q_km_min * 0.05, q_km_int[-1] + q_km_int[-1] * 0.2])
            plotter_q_km.add_labels(
                self.pth.get_label("V"),
                self.pth.get_label("q_km"),
            )
            plotter_q_km.add_text(
                V_int,
                q_km_int,
                min_position,
                self.pth.get_plot_text("q_km")[0],
            )
            plotter_q_km.set_legend()
            if save:
                plotter_q_km.save_figure(
                    f"q_km_H={round(self.altitude, 4)}", self.save_path
                )
            plotter_q_km.close_plot()

        return V_4, q_km_min

    def plot_q_km_q_ch_together(
        self,
        q_km,
        q_ch,
        V,
        min_q_km,
        min_q_ch,
        V_min_km,
        V_min_ch,
        file_name,
        text_side,
        save=True,
    ):
        if self.altitude >= 11:
            (V_int, q_km_int) = dh.prepare_data_for_plot(
                V, q_km, remove_first_element=True
            )
            (V_int, q_ch_int) = dh.prepare_data_for_plot(
                V, q_ch, remove_first_element=True
            )
        else:
            (V_int, q_km_int) = dh.prepare_data_for_plot(V, q_km)
            (V_int, q_ch_int) = dh.prepare_data_for_plot(V, q_ch)

        for type_name in self.TYPE_NAMES:
            plotter_q = plot(V_int, save_type=type_name, fun1=q_km_int)
            plotter_q.get_figure(self.pth.get_label_in_box("q_km")[0])
            plotter_q.add_labels(
                self.pth.get_label("V"),
                self.pth.get_label("q_km"),
            )
            plt.ylim([min_q_km - min_q_km * 0.25, q_km_int[-1] + q_km_int[-1] * 0.2])
            plotter_q.add_text(
                [0, V_min_km],
                [0, min_q_km],
                1,
                self.pth.get_plot_text("q_km")[0],
                side=text_side,
            )

            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax2.plot(
                V_int,
                q_ch_int,
                linestyle="--",
                color="green",
                label=self.pth.get_label_in_box("q_ch")[0],
            )
            ax2.plot(V_min_ch, min_q_ch, "o", color="orange")
            ax2.annotate(
                self.pth.get_plot_text("q_ch")[0] + f"{round(V_min_ch,3)}",
                xy=(V_min_ch, min_q_ch + (min_q_ch * 0)),
            )
            ax2.set_ylabel(self.pth.get_label("q_ch")[0])

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc=2)
            #            plotter_q_km.set_notation(2)
            if save:
                plotter_q.save_figure(file_name, self.save_path)
            plotter_q.close_plot()

    def plot_V_y_H(self, Vy_max, alts, H_pr, H_st, save=True):
        for type_name in self.TYPE_NAMES:
            plotter_V_y_H = plot(Vy_max, save_type=type_name, fun1=alts)
            plotter_V_y_H.get_figure(
                self.pth.get_label_in_box("V_y")[1],
            )
            plt.xlim(0, Vy_max[0] + 5)
            plt.ylim(0, alts[-1] + 2)
            plt.plot(0.5, H_pr, "o")
            plotter_V_y_H.add_labels(
                self.pth.get_label("V_y"),
                self.pth.get_label("H"),
            )

            plt.annotate(
                self.pth.get_plot_text("H")[1] + f"{round(H_pr,3)}",
                xy=(0.5, H_pr - (H_pr * 0.05)),
            )

            plotter_V_y_H.add_text(
                np.array([0]),
                np.array([H_st]),
                0,
                self.pth.get_plot_text("H")[0],
                add_value="y",
                text_location="up",
            )
            plotter_V_y_H.set_legend()
            if save:
                plotter_V_y_H.save_figure("V_y_H", self.save_path)
            plotter_V_y_H.close_plot()

    def plot_H_M(self, alts, M_min_P, M_max_P, M_min_dop, M_Vi_max, M_OGR, save=True):
        for type_name in self.TYPE_NAMES:
            try:
                no_limit_index = np.where(M_min_P == 0)[0][-1]
            except IndexError:
                no_limit_index = False

            if not isinstance(no_limit_index, bool):
                plotter_H_M = plot(
                    alts,
                    save_type=type_name,
                    fun1=M_min_dop,
                    fun2=M_Vi_max,
                    fun3=M_OGR,
                    fun4=M_max_P,
                )
                plotter_H_M.get_figure(
                    "$M_{Cy_{доп}}$",
                    "$M_{q_{max}}$",
                    "$M_{пред}$",
                    "$M_{max_{P}}$",
                    t_graph=True,
                )
                plotter_H_M.add_plot(
                    M_min_P[no_limit_index + 1 :],
                    alts[no_limit_index + 1 :],
                    "$M_{min_{P}}$",
                )
            else:
                plotter_H_M = plot(
                    alts,
                    save_type=type_name,
                    fun1=M_min_P,
                    fun2=M_max_P,
                    fun3=M_min_dop,
                    fun4=M_Vi_max,
                    fun5=M_OGR,
                )
                plotter_H_M.get_figure(
                    "$M_{min_{P}}$",
                    "$M_{max_{P}}$",
                    "$M_{Cy_{доп}}$",
                    "$M_{q_{max}}$",
                    "$M_{пред}$",
                    t_graph=True,
                )

            plt.plot([0, 1], [alts[-1], alts[-1]], "--k", linewidth=1)

            plotter_H_M.add_text(
                [0, 0.5],
                [alts[-1], alts[-1] + (0.02 * alts[-1])],
                1,
                str("$H_{пр}= %.2f \ км$" % (alts[-1])),
                add_value=None,
                marker_style="-",
            )

            plt.xlim(0, 1)
            plt.ylim(0, alts[-1] + 1)
            plt.xticks(np.arange(0, 1, 0.1))
            plotter_H_M.add_labels(
                self.pth.get_label("M"),
                self.pth.get_label("H"),
            )
            plt.legend(loc=2)
            if save:
                plotter_H_M.save_figure(f"H_M_flight_area", self.save_path)
                plotter_H_M.close_plot()

    def plot_q_ch_q_km(self, alts, q_km_min, q_ch_min, save=True):
        if any(np.isinf(q_ch_min)) or any(np.isinf(q_km_min)):
            del_pos = np.where(np.isinf(q_km_min))[0]
            q_km_min = np.delete(q_km_min, del_pos)
            q_ch_min = np.delete(q_ch_min, del_pos)
            alts = np.delete(alts, del_pos)

        H_int, q_km_min_int, q_ch_min_int = dh.prepare_data_for_plot(
            alts, q_km_min, q_ch_min / 1000
        )

        for type_name in self.TYPE_NAMES:
            plotter_q_ch_q_km = plot(
                H_int, save_type=type_name, fun1=q_km_min_int, fun2=q_ch_min_int
            )
            plotter_q_ch_q_km.get_figure(
                self.pth.get_label_in_box("q_km")[1],
                self.pth.get_label_in_box("q_ch")[1],
                t_graph=True,
            )
            plotter_q_ch_q_km.add_labels(
                self.pth.get_label("q_km") + self.pth.get_label("q_ch")[1],
                self.pth.get_label("H"),
            )
            plt.ylim(0, alts[-1] + 1)
            plotter_q_ch_q_km.set_legend()
            if save:
                plotter_q_ch_q_km.save_figure("q_km_q_ch", self.save_path)
            plotter_q_ch_q_km.close_plot()

    def plot_climb_param(
        self, t, alts, teta, Vy, V, save=True, file_name="climb_params"
    ):
        x_t = dh.sum_array(t)
        for type_name in self.TYPE_NAMES:

            plot_climb = plot(x_t, save_type=type_name, fun1=alts, fun2=teta, fun3=Vy)
            plot_climb.get_figure(
                "$H(t)[км]$", "$\\theta(t)\\,[град]$", "$V_y^*(t)\\,[м/с]$"
            )
            plot_climb.add_labels(
                "t [мин]", "$\\theta(t) [град]$, $V_y^*(t) [м/с]$, $H [км]$"
            )
            plot_climb.set_legend()
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.set_ylabel("V\\,[м/с]")
            ax2.plot(x_t, V, color="black", label="$V(t)\\,[м/с]$")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc=2)

            if save:
                plot_climb.save_figure(file_name, self.save_path)
            plot_climb.close_plot()

    def plot_L_m(self, t, L, m, file_name, save=True):
        x_t = dh.sum_array(t)
        s_L = dh.sum_array(L)
        s_m = dh.sum_array(m)
        for type_name in self.TYPE_NAMES:
            plot_L_m = plot(x_t, save_type=type_name, fun1=s_L)
            plot_L_m.get_figure("$L(t) [км]$")
            plot_L_m.add_labels("$t [мин]$", "$L [км]$")
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.plot(x_t, s_m, linestyle="--", color="green", label="$m_T(t) [кг]$")
            ax2.set_ylabel("$m_T [кг]$")

            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc=2)
            plot_L_m.set_notation(2)

            if save:
                plot_L_m.save_figure(file_name, self.save_path)
            plot_L_m.close_plot()

    def plot_H_M_profile(self, M, H, file_name, save=True):
        for type_name in self.TYPE_NAMES:
            plot_climb = plot(M, save_type=type_name, fun1=H)
            plot_climb.get_figure("$H(M)$")
            plot_climb.add_labels("$M$", "$H [км]$")
            plot_climb.set_legend()
            plt.xlim(0, 1)
            if save:
                plot_climb.save_figure(file_name, self.save_path)
            plot_climb.close_plot()

    def plot_L_H(
        self,
        H,
        L,
        save=True,
    ):

        for type_name in self.TYPE_NAMES:
            plot_L_H = plot(L, save_type=type_name, fun1=H)
            plot_L_H.get_figure("$H(L) [Км]$")
            plot_L_H.add_labels("$L [км]$", "$H [км]$")
            plot_L_H.set_legend()
            plt.xlim(0, L[-1] + 100)
            plt.ylim(0, np.max(H) + 2)
            if save:
                plot_L_H.save_figure("H_L_graph", self.save_path)
            plot_L_H.close_plot()

    def plot_cargo(self, L, m, save=True):
        for type_name in self.TYPE_NAMES:
            plot_cargo = plot(L, save_type=type_name, fun1=m)
            plot_cargo.get_figure("$m_{цн}(L)$")
            plot_cargo.add_labels("$L [км]$", "$m_{цн} [кг]$")
            plot_cargo.set_legend()
            if save:
                plot_cargo.save_figure("m_L_graph", self.save_path)
            plot_cargo.close_plot()

    def plot_turn(self, ny, omega, r, t, save=True):
        for type_name in self.TYPE_NAMES:
            plot_turn = plot(
                self.MACH,
                save_type=type_name,
                fun1=ny,
                fun2=omega * (10**2),
                fun3=r * (10**-3),
                fun4=t * (10**-1),
            )
            plot_turn.get_figure(
                "$n_{y_{вир}}$",
                "$\\omega_{вир}*10^{2}$",
                "$r_{вир} * 10^{-3}$",
                "$t_{вир} * 10^{-1}$",
            )
            plot_turn.add_labels("$M$", r"$n_{y},\, \omega [1/c],\, r [м],\, t[с]$")
            plot_turn.set_legend()
            if save:
                plot_turn.save_figure("turn_graph", self.save_path)
            plot_turn.close_plot()

    def plot_center_value(
        self, otn_s_go, otn_x_tpp, otn_x_tpz, xtpz_star, xtpp_star, S_star, save=True
    ):
        for type_name in self.TYPE_NAMES:
            plot_center = plot(
                otn_s_go,
                save_type=type_name,
                f1=otn_x_tpz,
                f2=otn_x_tpp,
            )
            plot_center.get_figure(
                "$\\bar{x}_{ТПЗ}(\\bar{S}_{го})$",
                "$\\bar{x}_{ТПП}(\\bar{S}_{го})$",
            )
            plt.plot([0, S_star], [xtpp_star, xtpp_star], color="k", linewidth=0.5)
            plt.plot([0, S_star], [xtpz_star, xtpz_star], color="k", linewidth=0.5)
            plt.plot([S_star, S_star], [xtpp_star, xtpz_star], color="k", linewidth=0.5)
            plt.plot([S_star], [xtpz_star], "ko")
            plt.plot([S_star], [xtpp_star], "ko")

            plot_center.add_text(
                [0, 0],
                [0, xtpp_star - (xtpp_star*0.1)],
                1,
                r"$\bar{x}_{тпп}(\bar{S}_{го}^*)=%.3f$" % (xtpp_star),
                add_value="",
                text_location="down",
                marker_style="-",
            )

            plot_center.add_text(
                [0, 0],
                [0, xtpz_star + xtpp_star * 0.1],
                1,
                r"$\bar{x}_{тпз}(\bar{S}_{го}^*)=%.3f$" % (xtpz_star),
                add_value="",
                text_location="up",
                marker_style="-",
            )

            plot_center.add_text(
                [S_star, S_star],
                [0, xtpp_star + ((xtpz_star - xtpp_star) / 2)],
                1,
                r"$\bar{S}_{го}^*=$",
                add_value="x",
                text_location="up",
                marker_style="-",
            )

            plot_center.add_labels("$\\bar{S}_{го}$", "$\\bar{x}_{Т}$")
            plot_center.set_legend()
            plt.xlim([0, otn_s_go[-1]])

            if save:
                plot_center.save_figure("xTP_graph", self.save_path)
            plot_center.close_plot()

    def plot_phi_bal(self, alts, mach_speeds, phi_bal, save=True):
        for type_name in self.TYPE_NAMES:
            plot_phi = plot(
                mach_speeds[0],
                save_type=type_name,
                fun1=phi_bal[0],
            )
            plot_phi.get_figure(
                "$\\varphi_{бал}(M,H=%s)$" % (alts[0]),
            )
            for i in range(1, len(alts)):
                plot_phi.add_plot(
                    mach_speeds[i], phi_bal[i], "$\\varphi_{бал}(M,H=%s)$" % (alts[i])
                )
            plot_phi.set_legend()
            plot_phi.add_labels("$M$", "$\\varphi_{бал}[град]$")
            if save:
                plot_phi.save_figure("phi_bal_graph", self.save_path)
            plot_phi.close_plot()

    def plot_phi_n(self, alts, mach_speeds, phi_n, save=True):
        for type_name in self.TYPE_NAMES:
            plot_phi_n = plot(
                mach_speeds[0],
                save_type=type_name,
                fun1=phi_n[0],
            )
            plot_phi_n.get_figure(
                "$\\varphi^{n}(M,H=%s)$" % (alts[0]),
            )
            for i in range(1, len(alts)):
                plot_phi_n.add_plot(
                    mach_speeds[i], phi_n[i], "$\\varphi^{n}(M,H=%s)$" % (alts[i])
                )
            plot_phi_n.set_legend()
            plot_phi_n.add_labels("$M$", "$\\varphi^{n}[град/ед.перег.]$")
            if save:
                plot_phi_n.save_figure("phi_n_graph", self.save_path)
            plot_phi_n.close_plot()

    def plot_ny_p(self, alts, mach_speeds, ny_p, ny_dops, save=True):
        for type_name in self.TYPE_NAMES:
            plot_phi_n = plot(
                mach_speeds[0],
                save_type=type_name,
                fun1=ny_p[0],
            )
            plot_phi_n.get_figure(
                "$n_{yp}(M,H=%s)$" % (alts[0]),
            )

            plt.plot(
                mach_speeds[0],
                ny_dops[0],
                "--",
                label="$n_{y_{доп}}(M,H=%s)$" % (alts[0]),
                color=plt.gca().lines[-1].get_color(),
            )
            for i in range(1, len(alts)):
                plot_phi_n.add_plot(
                    mach_speeds[i], ny_p[i], "$n_{yp}(M,H=%s)$" % (alts[i])
                )
                plt.plot(
                    mach_speeds[i],
                    ny_dops[i],
                    "--",
                    label="$n_{y_{доп}}(M,H=%s)$" % (alts[i]),
                    color=plt.gca().lines[-1].get_color(),
                )

            plot_phi_n.set_legend()
            plot_phi_n.add_labels("$M$", "$n_{yp}$")
            if save:
                plot_phi_n.save_figure("ny_p_graph", self.save_path)
            plot_phi_n.close_plot()

    def plot_xs_sigmas(self, x_f, x_H, x_tpz, sigma_n, M, save=True):
        for type_name in self.TYPE_NAMES:
            plotter = plot(
                M,
                save_type=type_name,
                fun1=x_f,
                fun2=x_H,
                fun3=x_tpz,
                fun4=sigma_n,
            )
            plotter.get_figure(
                r"$\bar{x}_F(M)$",
                r"$\bar{x}_H(M)$",
                r"$\bar{x}_{ТПЗ}(M)$",
                r"$\sigma_n(M)$",
            )

            plotter.set_legend()
            plotter.add_labels("$M$", "")
            if save:
                plotter.save_figure("xis_sigma", self.save_path)
            plotter.close_plot()

    def plot_aerodynamcis_data(self, mach, Cym, alpha_0, Cy_dop, Cy_alpha, Cx_m, A):
        for type_name in self.TYPE_NAMES:
            plotter = plot(
                    mach,
                    save_type=type_name,
                    f1=Cym*10, f2=alpha_0*10*10, f3=Cy_dop, f4=Cy_alpha/10, f5=Cx_m*10, f6=A,
                    )
            plotter.get_figure(
                    r'$C_{y_m} \times 10$',
                    r'$-\alpha_0\times 10^2$',
                    r'$C_{y_{доп}}$',
                    r'$C_{y}^{\alpha} \times 10^{-1}$',
                    r'$C_{x_{m}} \times 10$',
                    r'$A$',
                    )
            plotter.set_legend()
            plotter.add_labels("$M$", "") 
            plotter.save_figure("aero_data", self.save_path)
            plotter.close_plot()

    def plot_cy_cx(self, type1_cy_cx, type2_cy_cx, type3_cy_cx, type1_cy_alpha,
            type2_cy_alpha, type3_cy_alpha):

        for type_name in self.TYPE_NAMES:
            plotter = plot(
                    type1_cy_cx[1]*100,
                    save_type=type_name,
                    f1=type1_cy_cx[0],
                    )
            plotter.get_figure('1. $C_y(C_x)$') 
            plt.title("Режимы: 1. Взлетный, 2. Посадочный, \n3. Пробег с выпущенными интерцепторами.")
            plotter.add_plot(type2_cy_cx[1]*100, type2_cy_cx[0], '2. $C_y(C_x)$')
            plotter.add_plot(type3_cy_cx[1]*100, type3_cy_cx[0], '3. $C_y(C_x)$')

            plotter.add_plot(type1_cy_alpha[1], type1_cy_alpha[0], r'1. $C_y(\alpha)$')
            plotter.add_plot(type2_cy_alpha[1], type2_cy_alpha[0], r'2. $C_y(\alpha)$')
            plotter.add_plot(type3_cy_alpha[1], type3_cy_alpha[0], r'3. $C_y(\alpha)$')
            plotter.set_legend()
            plotter.add_labels(r"$C_{x} \times 10^{2}, \, \alpha \,[град]$", r'$C_y$')
            plotter.save_figure("aero_data_cx_cy_alpha", self.save_path)
            plotter.close_plot()

    def plot_tilda_Ces(self, tilda_Ce, mach, H):
        for type_name in self.TYPE_NAMES:
            plotter = plot(
                    mach,
                    save_type=type_name,
                    f1=tilda_Ce[0],
                    )
            plotter.get_figure(f'H={H[0]} км')
            for i, alt in enumerate(H[1:]):
                plotter.add_plot(mach, tilda_Ce[i], f'H={alt} км')
            plt.xlim([0.2, 1])
            plotter.set_legend()
            plotter.add_labels(r"$M$", r'$\tilde{Ce}$')
            plotter.save_figure("tilda_Ce", self.save_path)
            plotter.close_plot()

    def plot_tilda_P(self, tilda_P , mach, H):
        for type_name in self.TYPE_NAMES:
            plotter = plot(
                    mach,
                    save_type=type_name,
                    f1=tilda_P[0],
                    )
            plotter.get_figure(f'H={H[0]} км')
            for i, alt in enumerate(H[1:]):
                plotter.add_plot(mach, tilda_P[i], f'H={alt} км')
            plt.xlim([mach[0], mach[-1]])
            plotter.set_legend()
            plotter.add_labels(r"$M$", r'$\tilde{P}$')
            plotter.save_figure("tilda_P", self.save_path)
            plotter.close_plot()

    def plot_Ce_dr(self, Cedr, R):
        for type_name in self.TYPE_NAMES:
            plotter = plot(
                    R,
                    save_type=type_name,
                    f1=Cedr,
                    )
            plotter.get_figure(r'$\hat{Ce}_{др}(R)$')
            plotter.set_legend()
            plt.xlim([R[0], R[-1]])
            plt.ylim([min(Cedr)-(0.25*max(Cedr)), max(Cedr)])
            plotter.add_labels(r"$\bar{R}$", r'$\hat{Ce}_{др}$')
            plotter.save_figure("Ce_dr_R", self.save_path)
            plotter.close_plot()

    def plot_aero_data_elements(self, mach,
            SSC_K_go,
            Cy0_bgo,
            Cya_bgo,
            Cy_go_a_go,
            epsilon_a,
            m_z_bgo_w_z,
            n_v,
            otn_x_f_bgo,
            mz0_bgo,
            ):

        for type_name in self.TYPE_NAMES:
            plotter = plot(
                    mach,
                    save_type=type_name,
                    f0=SSC_K_go,
                    f1=Cy0_bgo,
                    f2=Cya_bgo/10,
                    f3=Cy_go_a_go/10,
                    f4=epsilon_a,
                    f5=-m_z_bgo_w_z/10,
                    f6=n_v,
                    f7=otn_x_f_bgo,
                    f8=-mz0_bgo,
                    )
            plotter.get_figure(
                    r'$K_{го}$',
                    r'$C_{{cy0}_{БГО}}$',
                    r'$C_{{y}_{БГО}}^\alpha \times 10^{-1}$',
                    r'$C_{{y}_{БГО}}^{\alpha_{ГО}}\times 10^{-1}$',
                    r'$\varepsilon_\alpha$',
                    r'$-m_{z_{БГО}}^{\bar{\omega}_z} \times 10^{-1}$',
                    r'$n_{в}$',
                    r'$\bar{x}_{F_{БГО}}$',
                    r'$-m_{{z0}_{БГО}}$',
                    )
            plotter.set_legend()
            plt.xlim([0.3, mach[-1]])
            plotter.add_labels(r'$M$', '')
            plotter.save_figure("aero_data_elements", self.save_path)
            plotter.close_plot()







