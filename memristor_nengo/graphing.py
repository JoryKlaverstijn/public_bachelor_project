import numpy as np
import math
import matplotlib.pyplot as plt
import random

exp_a = -0.093
exp_b = -0.53

def get_color(i):
    if i == 0:
        return [0, 0, 0]
    elif i == 1:
        return[63/255, 21/255, 133/255]
    elif i == 2:
        return[4/255, 15/255, 56/255]
    elif i == 3:
        return[26/255, 62/255, 191/255]
    elif i == 4:
        return[8/255, 255/255, 226/255]
    elif i == 5:
        return[86/255, 156/255, 16/255]
    elif i == 6:
        return[131/255, 227/255, 39/255]
    elif i == 7:
        return[140/255, 140/255, 140/255]
    elif i == 8:
        return[245/255, 163/255, 39/255]
    elif i == 9:
        return[245/255, 10/255, 10/255]

# The function that must equal zero
def func(V, Q, W):
    return 1 + pow(Q, 1/(exp_a+exp_b * V)) - pow(W, 1/(exp_a+exp_b * V))


# The derivative of the function
def der_func(V, Q, W):
    comp_1 = (exp_b * np.log(Q) * np.exp(np.log(Q)/ (exp_a + exp_b * V))) / pow((exp_a + exp_b * V), 2)
    comp_2 = (exp_b * np.log(W) * np.exp(np.log(W) / (exp_a + exp_b * V))) / pow((exp_a + exp_b * V), 2)
    return -comp_1 + comp_2


# The second derivative of the function
def sec_der_func(V, Q, W):
    P = exp_a + exp_b * V
    comp1 = pow(exp_b, 2) * pow(Q, 1/P) * pow(np.log(Q), 2)
    comp2 = pow(exp_b, 2) * pow(W, 1 / P) * pow(np.log(W), 2)
    return comp1 - comp2


# Halley method
def halley_method(V, Q, W):
    denominator = 2 * func(V, Q, W) * der_func(V, Q, W)
    numerator = 2 * pow(der_func(V, Q, W), 2) - func(V, Q, W) * sec_der_func(V, Q, W)
    return V - denominator / numerator


def plot_memristors_decrease():
    exp_a = -0.093
    exp_b = -0.53

    r_min = 200.0
    r_max = 2.3e8
    plots = []

    for V in range(1, 11):
        plot = []
        r_cur = 2.3e8
        plot.append(r_cur)
        for pulse in range(11):
            volt = float(V) / 10.0
            forw_exp = exp_a + exp_b * abs(volt)
            forw_n = np.power((r_cur - r_min) / r_max, 1.0 / forw_exp)
            new_res = r_min + r_max * np.power(forw_n + 1, forw_exp)
            plot.append(new_res)
            r_cur = new_res
        plots.append(plot)

    for i in range(10):
        col = get_color(i)
        plt.plot(plots[i],  color=col)
        plt.plot(plots[i], 'o', color=col, label=("+" + str((i + 1.0) / 10) + " V"), mfc='none')
    plt.legend(ncol=2,handleheight=1.6, labelspacing=0.05)
    plt.xlabel("Pulse number", fontsize=15)
    plt.ylabel("Resistance (Ω)", fontsize=15)
    plt.ylim((0, 2.5e8))

    plt.show()


def memristor_vs_normalized_plot(timesteps=100, volt=0.1):
    exp_a = -0.093
    exp_b = -0.53

    r_min = 200.0
    r_max = 2.3e8

    volt = 0.1

    plot_mem = []
    plot_lin = []

    r_cur = 2.3e8
    plot_mem.append(r_cur)
    for pulse in range(timesteps):
        forw_exp = exp_a + exp_b * abs(volt)
        forw_n = np.power((r_cur - r_min) / r_max, 1.0 / forw_exp)
        new_res = r_min + r_max * np.power(forw_n + 1, forw_exp)
        plot_mem.append(new_res)
        r_cur = new_res

    r_cur = 2.3e8
    r_des = 2.3e8
    plot_lin.append(r_cur)
    for pulse in range(timesteps):
        forw_exp = exp_a + exp_b * abs(volt)
        forw_n = np.power((r_cur - r_min) / r_max, 1.0 / forw_exp)
        new_res = r_min + r_max * np.power(forw_n + 1, forw_exp)
        r_des -= 1e6
        if new_res > r_des:
            r_cur = new_res
        plot_lin.append(r_cur)

    plt.plot(plot_mem, label="non-linearised")
    plt.plot(plot_lin, label="linearised")
    plt.xlabel("Pulse number", fontsize=15)
    plt.ylabel("Resistance (Ω)", fontsize=15)
    plt.legend(ncol=2, handleheight=2.4, labelspacing=0.05)
    plt.show()


def memristor_vs_halley(timesteps=100, volt=0.1):
    exp_a = -0.093
    exp_b = -0.53

    r_min = 200.0
    r_max = 2.3e8

    volt = 0.1

    plot_mem = []
    plot_lin = []

    r_cur = 1e8
    plot_mem.append(r_cur)
    for pulse in range(timesteps):
        forw_exp = exp_a + exp_b * abs(volt)
        forw_n = np.power((r_cur - r_min) / r_max, 1.0 / forw_exp)
        new_res = r_min + r_max * np.power(forw_n + 1, forw_exp)
        plot_mem.append(new_res)
        r_cur = new_res

    r_cur = 1e8
    r_des = 1e8
    plot_lin.append(r_cur)
    for pulse in range(timesteps):
        r_des -= 1e3
        Q = (r_cur - r_min) / r_max
        W = (r_des - r_min) / r_max

        volt = 1.0
        for _ in range(5):
            volt = halley_method(volt, Q, W)

        print(volt, r_cur-r_des)
        if volt < 0.1:
            volt = 0.0
        if volt > 10.0:
            volt = 10.0

        if volt > 0.1:
            forw_exp = exp_a + exp_b * abs(volt)
            forw_n = np.power((r_cur - r_min) / r_max, 1.0 / forw_exp)
            r_cur = r_min + r_max * np.power(forw_n + 1, forw_exp)
        plot_lin.append(r_cur)

    plt.plot(plot_mem, label="non-linearised")
    plt.plot(plot_lin, label="linearised")
    plt.xlabel("Pulse number", fontsize=15)
    plt.ylabel("Resistance (Ω)", fontsize=15)
    plt.legend(ncol=2, handleheight=2.4, labelspacing=0.05)
    plt.show()


def input_nodes_figure():
       fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
       y1 = [-2, 2, -2]
       y2 = [2, 2, -2]
       y3 = [2, -2, 2]
       x = [0, 1, 1.00001, 2, 2.00001, 3]

       ax1.plot(x, np.repeat(y1, 2), color="black", linewidth=7.0)
       ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
       ax1.set_ylabel("Node output")

       ax2.plot(x, np.repeat(y2, 2), color="black", linewidth=7.0)
       ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
       ax2.set_ylabel("Node output")

       ax3.plot(x, np.repeat(y3, 2), color="black", linewidth=7.0)
       ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
       ax3.set_ylabel("Node output")

       plt.xlabel("Time")


       plt.show()

       fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

       s1 = np.arange(21, 40)
       s2 = np.arange(1, 40)
       s3 = np.concatenate((np.arange(1, 20), np.arange(40, 60)))

       plt.setp(ax1.get_xticklabels(), visible=False)
       plt.setp(ax1.get_yticklabels(), visible=False)
       plt.setp(ax2.get_xticklabels(), visible=False)
       plt.setp(ax2.get_yticklabels(), visible=False)
       plt.setp(ax3.get_xticklabels(), visible=False)
       plt.setp(ax3.get_yticklabels(), visible=False)
       ax1.set_xlim([0, 60])
       ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
       ax1.tick_params(axis=u'both', which=u'both', length=0)
       ax1.eventplot(s1, color="black")
       ax2.set_xlim([0, 60])
       ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
       ax2.eventplot(s2, color="black")
       ax3.set_xlim([0, 60])
       ax3.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
       ax3.eventplot(s3, color="black")

       plt.xlabel("Time")
       plt.show()


