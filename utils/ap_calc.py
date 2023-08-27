import numpy as np
import matplotlib.pyplot as plt


def P_R_calc(tp_fp_mass, total_num):
    tp_cum = 0
    fp_cum = 0
    P_mass = []
    R_mass = []
    for pred in tp_fp_mass:
        if pred == 1:
            tp_cum += 1
        else:
            fp_cum += 1
        P = tp_cum / (tp_cum + fp_cum)
        R = tp_cum / total_num

        P_mass.append(P)
        R_mass.append(R)

    return P_mass, R_mass


def find_max_P(R_P_points):
    max_p = R_P_points[0][1]
    for i, p in enumerate(R_P_points):
        if p[1] > max_p:
            max_p = p[1]
    return max_p


def calc_AP(P, R, approx_points_num=11):
    unique_P = []
    unique_R = np.unique(R)

    for rec in unique_R:
        r_idxs = [i for i in range(len(R)) if R[i] == rec]
        p_max = max([P[i] for i in r_idxs])
        unique_P.append(p_max)

    AP = 0

    delta = 1.0 / approx_points_num
    for i in range(approx_points_num):
        r_tek = delta * i
        points_right = [(unique_R[j], unique_P[j]) for j in range(len(unique_R)) if unique_R[j] >= r_tek]
        if len(points_right):
            AP += delta * (find_max_P(points_right))

    return AP


def print_P_R_table(tp_fp_mass, total_num, confs=None, col_size=15):
    header = f"{'Preds.':^{col_size}}|{'Conf.':^{col_size}}|{'Matches':^{col_size}}|{'Cumulative TP':^{col_size}}|{'Cumulative FP':^{col_size}}|{'P':^{col_size}}|{'R':^{col_size}}"
    print(header)
    tp_cum = 0
    fp_cum = 0

    for i, pred in enumerate(tp_fp_mass):
        print("-" * (7 * (col_size + 1)))

        if pred == 1:
            tp_cum += 1
            matches = 'TP'
        else:
            fp_cum += 1
            matches = 'FP'
        P = tp_cum / (tp_cum + fp_cum)
        R = tp_cum / total_num
        if confs:
            conf = confs[i]
        else:
            conf = "-"
        line = f"{i + 1:^{col_size}}|{conf:^{col_size}}|{matches:^{col_size}}|{tp_cum:^{col_size}}|{fp_cum:^{col_size}}|{P:^{col_size}.3}|{R:^{col_size}.3}"
        print(line)


def AP_plot(P, R, approx_points_num=11):
    fig, ax = plt.subplots()

    unique_P = []
    unique_R = np.unique(R)

    for rec in unique_R:
        r_idxs = [i for i in range(len(R)) if R[i] == rec]
        p_max = max([P[i] for i in r_idxs])
        unique_P.append(p_max)

    ax.scatter(unique_R, unique_P)
    ax.set_xlabel("R")
    ax.set_ylabel("P")

    # build approx curve

    approx_R = []
    approx_P = []

    delta = 1.0 / approx_points_num
    for i in range(approx_points_num):
        r_tek = delta * i
        approx_R.append(r_tek)
        points_right = [(unique_R[j], unique_P[j]) for j in range(len(unique_R)) if unique_R[j] >= r_tek]
        if len(points_right):
            approx_P.append(find_max_P(points_right))
        else:
            approx_P.append(0)

    approx_R_line = []
    approx_P_line = []
    for i in range(len(approx_R)):
        if i != len(approx_R) - 1:
            approx_P_line.append(approx_P[i])
            approx_R_line.append(approx_R[i])
            if approx_P[i + 1] < approx_P[i]:
                approx_P_line.append(approx_P[i + 1])
                approx_R_line.append(approx_R[i])

    ax.plot(approx_R_line, approx_P_line, c="red")

    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.1)

    ax.set_title(f"Зависимость Precision от Recall")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    tp_fp_mass = [1, 1, 1, 0, 1, 0, 1]
    P, R = P_R_calc(tp_fp_mass, 7)

    print("precision\n", P)
    print("recall\n", R)

    AP_plot(P, R)

    print(f"AP = {calc_AP(P, R)}")

    print_P_R_table(tp_fp_mass, 12)
