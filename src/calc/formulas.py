import numpy as np
import warnings
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)


class Formulas:
    def v_speed(mach, speed_of_sound):
        return mach * speed_of_sound

    def q_dynamic_pressure(v, air_density):
        return (air_density * (v) ** 2) / 2

    def C_y_n_lift_coefficient(otn_M, ps, q):
        return (otn_M * ps * 10) / q

    def C_x_n_drag_coefficient(C_x_m, A, C_y_n, C_y_m):
        return C_x_m + A * (C_y_n - C_y_m) ** 2

    def K_n_lift_to_drag_ratio(C_y_n, C_x_n):
        K_n = np.nan_to_num(C_y_n / C_x_n)
        return K_n

    def P_potr_equation(otn_M, M0, g, K_n):
        return otn_M * M0 * g / K_n

    def P_rasp_equation(otn_P_0, M0, g, tilda_P, alt, p_h_11, p_h):
        if alt >= 11:
            return otn_P_0 * M0 * g * tilda_P * (p_h / p_h_11)
        else:
            return otn_P_0 * M0 * g * tilda_P

    def n_x_equation(otn_M, M0, g, P_rasp, P_potr):
        return (P_rasp - P_potr) / (otn_M * M0 * g)

    def V_y_equation(V, n_x):
        return V * n_x

    def otn_R_equation(P_rasp, P_potr):
        return P_potr / P_rasp

    def q_ch_hour_consumption(Ce_0, tilda_Ce, tilda_Ce_dr, P_potr):
        return Ce_0 * tilda_Ce * tilda_Ce_dr * P_potr

    def q_km_range_consumption(q_ch, V):
        return q_ch / (3.6 * V)

    def M_V_i_max(V_i_max, sqrt_delta, a_H):
        return (V_i_max * sqrt_delta) / (3.6 * a_H)

    def find_M_nab(M_0, M2, H_nab, Vk):
        M_nab = np.array([M_0])
        for i, _ in enumerate(M2):
            if i > 0 and i < len(H_nab) - 1:
                M_nab = np.append(M_nab, M2[i])
        return np.append(M_nab, Vk)

    def find_M_des(M_0, M1, H_des):
        M_des = np.array([])
        for i, M1_value in enumerate(M1):
            if i < len(H_des) - 1:
                M_des = np.append(M_des, M1_value)
        M_des = np.append(M_des, M_0)
        return np.flip(M_des)

    def find_H_nab(alts, H_k, Vy_max):
        valid_pos = np.array([i for i, val in enumerate(Vy_max) if val > 0.1])
        H = np.array(
            [val for i, val in enumerate(alts) if val <= H_k and i in valid_pos]
        )
        H = np.array([val for i, val in enumerate(H) if not np.mod(val, 1)])
        return H

    def find_Vk(H_nab, V_4):
        return np.array([V_4[i] for i, _ in enumerate(V_4) if i == len(H_nab) - 1])

    def dVdH_equation(V_nab, H_nab):
        dVdH = np.array([])
        for i in range(0, len(H_nab) - 1):
            dVdH = np.append(
                dVdH, (V_nab[i + 1] - V_nab[i]) / (H_nab[i + 1] - H_nab[i])
            )
        dVdH = np.append(dVdH, 0)
        return dVdH

    def k_equation(V, dVdH, g):
        return 1 / (1 + (V / g) * dVdH)

    def teta_nab_equation(n_x, k):
        return n_x * k * 57.3

    def v_y_nab_equation(Vy_max, k):
        return Vy_max * k

    def H_energy_equation(H, V, g):
        return H + ((V) ** 2 / (2 * g))

    def delta_H_energy_equation(H_energy):
        delta_H_e = np.array([])
        for H_index, H_value in enumerate(H_energy):
            if H_index == len(H_energy) - 1:
                break
            else:
                delta_H_e = np.append(delta_H_e, H_energy[H_index + 1] - H_value)
        return np.append(delta_H_e, 0)

    def n_x_avg_equation(n_x):
        n_x_avg = np.array([])
        for i, _ in enumerate(n_x):
            if i == len(n_x) - 1:
                break
            else:
                n_x_avg = np.append(n_x_avg, 0.5 * ((1 / n_x[i]) + (1 / n_x[i + 1])))
        return np.append(n_x_avg, 0)

    def v_y_avg_equation(V_y):
        v_y_avg = np.array([])
        for i, _ in enumerate(V_y):
            if i == len(V_y) - 1:
                break
            else:
                v_y_avg = np.append(v_y_avg, 0.5 * ((1 / V_y[i]) + (1 / V_y[i + 1])))
        return np.append(v_y_avg, 0)

    def CeP_Vy_avg_equation(Ce, P, Vy_max):
        CeP_Vy_avg = np.array([])
        for i, _ in enumerate(Ce):
            if i == len(Ce) - 1:
                break
            else:
                CeP_Vy_avg = np.append(
                    CeP_Vy_avg,
                    0.5
                    * (
                        ((Ce[i] * P[i]) / Vy_max[i])
                        + (Ce[i + 1] * P[i + 1]) / Vy_max[i + 1]
                    ),
                )
        return np.append(CeP_Vy_avg, 0)

    def L_nab_equation(delta_H_e, nx_avg):
        return nx_avg * (delta_H_e / 1000)

    def t_nab_equation(delta_H_e, vy_avg):
        return vy_avg * (delta_H_e / 60)

    def m_t_equation(delta_H_e, CeP_Vy_avg):
        return CeP_Vy_avg * (delta_H_e / 3600)

    def otn_m_t_nab_equation(m_t, m0):
        return m_t / m0

    def otn_m_t_kr_equation(
        otn_m_ch,
        otn_m_sn,
        otn_m_t_nab,
        otn_m_t_snp,
        otn_m_t_anz,
        otn_m_t_pr,
    ):
        return (
            1
            - otn_m_ch
            - otn_m_sn
            - otn_m_t_nab
            - otn_m_t_snp
            - otn_m_t_anz
            - otn_m_t_pr
        )

    def otn_m_t_kr_mode_equation(
        otn_m_t_max,
        otn_m_t_nab,
        otn_m_t_CH,
        otn_m_t_anz,
        otn_m_t_pr,
    ):
        return otn_m_t_max - otn_m_t_nab - otn_m_t_CH - otn_m_t_anz - otn_m_t_pr

    def T_kr_equation(K_gp, g, Ce, otn_m_t_nab, otn_m_t_pr, otn_m_t_kr):
        return ((60 * K_gp) / (g * Ce)) * np.log(
            (1 - otn_m_t_nab - otn_m_t_pr) / (1 - otn_m_t_kr - otn_m_t_nab - otn_m_t_pr)
        )

    def L_kr_equation(V, K_gp, g, Ce, otn_m_t_nab, otn_m_t_pr, otn_m_t_kr, otn_m_vz=1):
        return ((3.6 * V * K_gp) / (g * Ce)) * np.log(
            (otn_m_vz - otn_m_t_nab - otn_m_t_pr)
            / (otn_m_vz - otn_m_t_kr - otn_m_t_nab - otn_m_t_pr)
        )

    def L_kr_equation_mode2(V, K_gp, g, Ce, otn_m_t_nab, otn_m_t_pr, otn_m_t_kr):
        return ((3.6 * V * K_gp) / (g * Ce)) * np.log(
            (1.5 - otn_m_t_nab - otn_m_t_pr)
            / (1.5 - otn_m_t_kr - otn_m_t_nab - otn_m_t_pr)
        )

    def otn_m_kkr_equation(otn_m_t_nab, otn_m_t_pr, otn_m_t_kr):
        return 1 - otn_m_t_nab - otn_m_t_pr - otn_m_t_kr

    def Ro_kr_equation(otn_m_kkr, Ps, Cy_gp, V):
        return (2 * otn_m_kkr * Ps * 10) / (Cy_gp * V ** 2)

    def V_rotate_equation(Ps, otn_P_to, alpha_r, ro_0, Cy_r):
        return math.sqrt(
            (20 * Ps * (1 - 0.9 * otn_P_to * math.sin(alpha_r))) / (ro_0 * Cy_r)
        )

    def Cp_equation(otn_P_to, f_p):
        return 0.9 * otn_P_to - f_p

    def bp_equation(Cx_p, f_p, Cy_p, ro_0, Ps):
        return (Cx_p - f_p * Cy_p) * (ro_0) / (2 * Ps * 10)

    def L_takeoff_equation(g, bp, Cp, V_r):
        return (1 / (2 * g * bp)) * np.log((Cp / (Cp - bp * V_r ** 2)))

    def V2_equation(V_r):
        return 1.1 * V_r

    def hat_V_avg_equation(V2, V_r):
        return math.sqrt((V2 ** 2 + V_r ** 2) / 2)

    def hat_nx_avg_equation(otn_P_to, Cx_r, ro_0, V_avg, Ps):
        return otn_P_to - ((Cx_r * ro_0 * V_avg ** 2) / (Ps * 20))

    def L_vuv_equation(nx_avg, V2, V_r, g, H_vzl):
        return (1 / nx_avg) * (((V2 ** 2 - V_r ** 2) / (2 * g)) + H_vzl)

    def otn_m_landing_equation(otn_m_kkr, otn_m_t_snp):
        return otn_m_kkr - otn_m_t_snp

    def v_touch_equation(otn_m_la, Ps, Cy_touch, ro_0):
        return math.sqrt((2 * otn_m_la * Ps * 10) / (Cy_touch * ro_0))

    def P_rev_equation(otn_P_0, n_rev, n_eng, m_la):
        return 0.4 * otn_P_0 * m_la * (n_rev / n_eng)

    def otn_P_rev_equation(p_rev, m_la, g):
        return (p_rev) / (m_la * g)

    def a_n_equation(otn_p_rev, f_la):
        return -otn_p_rev - f_la

    def b_n_equation(ro_0, otn_m_la, Ps, Cx_run, f_la, Cy_run):
        return ((ro_0) / (otn_m_la * Ps * 20)) * (Cx_run - f_la * Cy_run)

    def L_run_equation(g, b_n, a_n, V_touch):
        return (1 / (2 * g * b_n)) * np.log((a_n - b_n * V_touch ** 2) / (a_n))

    def Cy_landing_equation(Cy_touch):
        return 0.7 * Cy_touch

    def V_flare_equation(otn_m_la, Ps, Cy_la, ro_0):
        return math.sqrt((2 * otn_m_la * Ps * 10) / (Cy_la * ro_0))

    def K_landing_equation(Cy_la, Cx_la):
        return Cy_la / Cx_la

    def L_vup_equation(K_la, H_la, V_flare, V_touch, g):
        return K_la * (H_la + (V_flare ** 2 - V_touch ** 2) / (2 * g))

    def L_landing_equation(L_prob, L_vup):
        return L_prob + L_vup

    def otn_m_plane_equation(otn_m_t):
        return 1 - 0.5 * otn_m_t

    def n_y_equation(Cy_dop, Cy_gp):
        return Cy_dop / Cy_gp

    def n_y_p_equation(Cy_gp, Cy_m, otn_P, Cxm, A):
        return (1 / Cy_gp) * (Cy_m + ((otn_P * Cy_gp - Cxm) / A) ** (1 / 2))

    def otn_P_equation(P_rasp, m, g):
        return P_rasp / (m * g)

    def turn_mach(M_min_t, M_max_t, Mfly):
        M_ready = np.arange(min(Mfly), max(Mfly) + 0.01, 0.01)
        return np.array([val for val in M_ready if val > M_min_t and val < M_max_t])

    def turn_omega_equation(g, V, ny):
        return (g / V) * ((ny ** 2 - 1) ** (1 / 2))

    def turn_radius_equation(V, omega):
        return V / omega

    def turn_time_equation(r, V):
        return (2 * math.pi * r) / (V)

    def otn_x_tpp_equation(
        mz0_bgo, otn_xf_bgo, Cy_bgo, Cy_go, otn_S_go, K_go, otn_L_go
    ):
        return (-mz0_bgo + otn_xf_bgo * Cy_bgo + Cy_go * otn_S_go * K_go * otn_L_go) / (
            Cy_bgo
        )

    def Cy_bgo_equation(Cy0_bgo, Cya_bgo, alpha):
        return Cy0_bgo + Cya_bgo ** 2 * alpha

    def Cy_go_equation(Cya_go_go, alpha, epsilon_alpha, phi_ef):
        return Cya_go_go * (alpha * (1 - epsilon_alpha) + phi_ef)

    def phi_ef_equation(phi_ust, n_v, delta_max):
        return phi_ust + n_v * delta_max

    def otn_x_tpz_equation(otn_x_n, sigma_nmin):
        return otn_x_n + sigma_nmin

    def otn_x_n_equaiton(otn_x_f, m_otn_omega_z_z, mu):
        return otn_x_f - (m_otn_omega_z_z / mu)

    def mu_equation(Ps, Ro, g, b_a):
        return (2 * Ps * 10) / (Ro * g * b_a)

    def m_otn_omega_z_z_go_equation(Cya_go_go, otn_S_go, otn_L_go, K_go):
        return -Cya_go_go * otn_S_go * (otn_L_go ** 2) * (K_go) ** (1 / 2)

    def otn_x_f_equation(otn_x_f_bgo, delta_otn_x_f):
        return otn_x_f_bgo + delta_otn_x_f

    def delta_otn_x_f_equation(
        Cya_go_go, Cy_a, epsilon_alpha, otn_S_go, otn_L_go, K_go
    ):
        return ((Cya_go_go) / (Cy_a)) * (1 - epsilon_alpha) * otn_S_go * otn_L_go * K_go

    def m_z_otn_omega_z_equation(m_otn_omega_z_z_bgo, m_otn_omega_z_z_go):
        return m_otn_omega_z_z_bgo + m_otn_omega_z_z_go

    def otn_x_t_equation(otn_xtpz, otn_xtpp):
        return 0.5 * (otn_xtpz + otn_xtpp)

    def sigma_n_equation(otn_x_T, otn_x_F, mz_otn_w_z, mu):
        return otn_x_T - otn_x_F + (mz_otn_w_z / mu)

    def phi_bal_equation(m_z_0, m_z_Cy, Cy_gp, m_z_delta, otn_L_go, phi_ust, n_v):
        return -(m_z_0 + m_z_Cy * Cy_gp) / (m_z_delta * (1 + (m_z_Cy / otn_L_go))) + (
            phi_ust / n_v
        )

    def phi_n_equation(Cy_gp, sigma_n, m_z_delta):
        return -57.3 * ((Cy_gp * sigma_n) / m_z_delta)

    def nyp_equation(phi_max, phi_ust, phi_bal, phi_n):
        return 1 + ((phi_max + phi_ust - phi_bal) / phi_n)

    def m_z_0_equation(
        mz0_bgo, otn_S_go, otn_L_go, K_go, Cy_alpha_go_go, alpha_0, epsilon_alpha
    ):
        return mz0_bgo - otn_S_go * otn_L_go * K_go * Cy_alpha_go_go * alpha_0 * (
            1 - epsilon_alpha
        )

    def m_z_delta_equation(Cy_alpha_go_go, otn_S_go, otn_L_go, K_go, n_v):
        return -Cy_alpha_go_go * otn_S_go * otn_L_go * K_go * n_v

    def m_z_Cy_equation(otn_x_T, otn_x_F):
        return otn_x_T - otn_x_F
