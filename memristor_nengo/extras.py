import datetime
import os

import matplotlib.pyplot as plt
import nengo
import numpy as np
from nengo.processes import Process
from nengo.params import NdarrayParam, NumberParam
from nengo.utils.matplotlib import rasterplot
import tensorflow as tf
import time
import pickle

# Added Jory
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
from mnist.mnist import convert_mOja_img

# Plots all pulses of neuron between certain time range
def plot_neural_spiking(probe, trange):
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    rasterplot(trange, probe, ax)
    ax.set_ylabel('Neuron')
    ax.set_xlabel('Time (s)')
    fig.get_axes()[0].annotate("Pre" + " neural activity", (0.5, 0.94),
                               xycoords='figure fraction', ha='center', fontsize=20)
    return fig


# Plots the frequency for all neurons
def plot_neural_frequency(probe, trange):
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    n_neurons = len(probe[0])

    delta_t = trange[1] - trange[0]

    # Get all time steps at which neurons have spiked for each neuron
    spike_timestamps = []
    for n in range(n_neurons):
        timestamps_tmp = []
        t = 0
        for spikes in probe:
            if spikes[n] > 0:
                timestamps_tmp.append(t)
            t += delta_t
        spike_timestamps.append(timestamps_tmp)

    # Find the frequencies of spiking for each neuron
    frequencies = []
    for n in range(n_neurons):
        frequency = [0.0]
        for t in range(1, len(spike_timestamps[n])):
            frequency.append(1/(spike_timestamps[n][t]-spike_timestamps[n][t-1]))
        frequencies.append(frequency)

    # Remove first frequency value
    for n in range(n_neurons):
        if len(spike_timestamps[n]) == 0:
            spike_timestamps[n].append(0)
        spike_timestamps[n].pop(0)
        frequencies[n].pop(0)

    # Plot the frequencies with different line types
    plots = []
    labels = []
    for n in range(n_neurons):
        lb = f"Neuron {n}"
        pl, = plt.plot(spike_timestamps[n], frequencies[n], label=lb)
        plots.append(pl)
        labels.append(lb)

    # Make a slider to configure gaussian filter
    axsig = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='white')
    ssig = Slider(axsig, 'Gaussian filter: sigma', 1, 20, valinit=1, valstep=1)

    # Update every plot if sigma has changed
    def update(val):
        sig = int(ssig.val)
        for n in range(n_neurons):
            if len(spike_timestamps[n]) > 0:
                ydat = gaussian_filter1d(frequencies[n], sigma=sig)
                pl = plots[n]
                pl.set_ydata(ydat)
                fig.canvas.draw_idle()

    ssig.on_changed(update)

    # Add some labels
    ax.legend(bbox_to_anchor=(-0.1, 1.08))
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Frequency of Neuron Spiking')

    # Return the slider (otherwise it gives errors)
    return fig, ssig


# Plots a heatmap of the final weights (interactive)
def plot_weights_interactive(probe, trange, cmap="jet"):
    fig, ax = plt.subplots(1)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.set_cmap(cmap)
    ax.set_title('Heat Map of Weights')
    img = ax.imshow(probe[-1], cmap='jet', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=('white'))
    timestep_slider = Slider(axtime, 'time (s)', 0.0, trange[-2], trange[-2], valstep=trange[0])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Specify the time of the weights
    def update_depth(val):
        idx = np.floor(timestep_slider.val*1/trange[0])
        img.set_data(probe[int(idx)])

    timestep_slider.on_changed(update_depth)

    return fig, timestep_slider


def plot_weights_video(probe, trange, cmap="jet"):
    fig, ax = plt.subplots(1)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.set_cmap(cmap)
    ax.set_title('Heat Map of Weights')
    img = ax.imshow(probe[-1], cmap='jet', interpolation='nearest')
    fig.colorbar(img, ax=ax)
    axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=('white'))
    timestep_slider = Slider(axtime, 'time (s)', 0.0, trange[-2], trange[-2], valstep=trange[0])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Specify the time of the weights
    def update_depth(val):
        idx = np.floor(timestep_slider.val*1/trange[0])
        img.set_data(probe[int(idx)])

    timestep_slider.on_changed(update_depth)
    pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))

    return fig, timestep_slider


def plot_tuning_curves(eval_points, activities):
    fig, ax = plt.subplots(1)
    plt.plot(eval_points, activities)
    plt.ylabel("Firing rate (Hz)")
    plt.xlabel("Input scalar, x")
    plt.title("Tuning curves")

    return fig


def plot_trange(y, trange, xlabel='x', ylabel='y', title='title'):
    fig,ax = plt.subplots(1)
    plt.plot(trange, y)
    plt.ylabel(ylabel)
    plt.xlabel(ylabel)
    plt.title(title)

def plot_memristors(posmem, negmem, trange):
    samples = len(posmem)
    neurons = len(posmem[0])

    fig, axs = plt.subplots(neurons, neurons)
    #fig.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    fig.tight_layout()

    min_val = np.inf
    max_val = -np.inf
    posmems = []
    negmems = []
    for x in range(neurons):
        for y in range(neurons):
            cur_pos_neuron = []
            cur_neg_neuron = []
            for s in range(samples):
                if posmem[s][x][y] < min_val:
                    min_val = posmem[s][x][y]
                if negmem[s][x][y] < min_val:
                    min_val = posmem[s][x][y]
                if posmem[s][x][y] > max_val:
                    max_val = posmem[s][x][y]
                if negmem[s][x][y] > max_val:
                    max_val = posmem[s][x][y]
                cur_pos_neuron.append(posmem[s][x][y])
                cur_neg_neuron.append(negmem[s][x][y])
            posmems.append(cur_pos_neuron)
            negmems.append(cur_neg_neuron)

    print("min and max val:", min_val, max_val)

    for x in range(neurons):
        for y in range(neurons):
            axs[x, y].plot(trange,posmems[x + y * neurons],  'tab:red')
            axs[x, y].plot(trange, negmems[x + y * neurons], 'tab:green')
            axs[x, y].set_title(f'{x}, {y}', fontdict=dict(fontsize=10))
            axs[x, y].set_ylim(min_val*0.9, max_val*1.1)


    for ax in axs.flat:
        # ax.set(xlabel='Time (s)', ylabel='R (Ohm)')
        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('R (Ohm)', fontsize=20)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)

    for ax in axs.flat:
        ax.label_outer()

    return fig


# Makes a heatmap of the weights from each input neuron to each output neuron
def heatmap_onestep(probe, t=-1, title="Weights after learning"):
    if probe.shape[1] > 100:
        print("Too many neurons to generate heatmap")
        return

    cols = 10
    rows = int(probe.shape[1] / cols)
    if int(probe.shape[1] / cols) % cols != 0 or probe.shape[1] < cols:
        rows += 1

    plt.set_cmap('jet')
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, 1.75 * rows), dpi=100)
    for i, ax in enumerate(axes.flatten()):
        try:
            ax.matshow(probe[t, i, ...].reshape((28, 28)))
            ax.set_title(f"N. {i}")
            ax.set_yticks([])
            ax.set_xticks([])
        except:
            ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()

    return fig


# Makes a video of the heatmap of the weights during learning
def generate_heatmap(probe, folder, sampled_every, num_samples=None):
    if probe.shape[1] > 100:
        print("Too many neurons to generate heatmap")
        return

    try:
        os.makedirs(folder + "tmp")
    except FileExistsError:
        pass

    num_samples = num_samples if num_samples else probe.shape[0]
    step = int(probe.shape[0] / num_samples)

    print("Saving Heatmaps ...")
    for i in range(0, probe.shape[0], step):
        print(f"Saving {i} of {num_samples} images", end='\r')
        fig = heatmap_onestep(probe, t=i, title=f"t={np.rint(i * sampled_every)}")
        fig.savefig(folder + "tmp" + "/" + str(i).zfill(10) + ".png", transparent=True, dpi=100)
        plt.close()

    print("Generating Video from Heatmaps ...")
    os.system(
        "ffmpeg "
        "-pattern_type glob -i '" + folder + "tmp" + "/" + "*.png' "
                                                           "-c:v libx264 -preset veryslow -crf 17 "
                                                           "-tune stillimage -hide_banner -loglevel warning "
                                                           "-y -pix_fmt yuv420p "
        + folder + "weight_evolution" + ".mp4")
    if os.path.isfile(folder + "weight_evolution" + ".mp4"):
        os.system("rm -R " + folder + "tmp")


def pprint_dict(d, level=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("\t" * level, f"{k}:")
            pprint_dict(v, level=level + 1)
        else:
            print("\t" * level, f"{k}: {v}")


def setup():
    import sys

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

    # for nengo GUI
    sys.path.append("")
    # for rosa
    sys.path.append("../THOMAS")

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class PresentInputWithPause(Process):
    """Present a series of inputs, each for the same fixed length of time.

    Parameters
    ----------
    inputs : array_like
        Inputs to present, where each row is an input. Rows will be flattened.
    presentation_time : float
        Show each input for this amount of time (in seconds).
    """

    inputs = NdarrayParam("inputs", shape=("...",))
    presentation_time = NumberParam("presentation_time", low=0, low_open=True)
    pause_time = NumberParam("pause_time", low=0, low_open=True)

    def __init__(self, inputs, presentation_time, pause_time, **kwargs):
        self.inputs = inputs
        self.presentation_time = presentation_time
        self.pause_time = pause_time

        super().__init__(
            default_size_in=0, default_size_out=self.inputs[0].size, **kwargs
        )

    def make_step(self, shape_in, shape_out, dt, rng, state):
        assert shape_in == (0,)
        assert shape_out == (self.inputs[0].size,)

        n = len(self.inputs)
        inputs = self.inputs.reshape(n, -1)
        presentation_time = float(self.presentation_time)
        pause_time = float(self.pause_time)

        def step_presentinput(t):
            total_time = presentation_time + pause_time
            i = int((t - dt) / total_time + 1e-7)
            ti = t % total_time
            return np.zeros_like(inputs[0]) if ti > presentation_time else inputs[i % n]

        return step_presentinput


class Sines(Process):

    def __init__(self, period=4, **kwargs):
        super().__init__(default_size_in=0, **kwargs)

        self.period = period

    def make_step(self, shape_in, shape_out, dt, rng, state):
        # iteratively build phase shifted sines
        s = "lambda t: ("
        phase_shift = (2 * np.pi) / shape_out[0]
        for i in range(shape_out[0]):
            s += f"np.sin( 1 / {self.period} * 2 * np.pi * t + {i * phase_shift}),"
        s += ")"
        signal = eval(s)

        def step_sines(t):
            return signal(t)

        return step_sines


class SwitchInputs(Process):
    def __init__(self, pre_switch, post_switch, switch_time, **kwargs):
        assert issubclass(pre_switch.__class__, Process) and issubclass(post_switch.__class__, Process), \
            f"Expected two nengo Processes, got ({pre_switch.__class__},{post_switch.__class__}) instead"

        super().__init__(default_size_in=0, **kwargs)

        self.switch_time = switch_time
        self.preswitch_signal = pre_switch
        self.postswitch_signal = post_switch

    def make_step(self, shape_in, shape_out, dt, rng, state):
        preswitch_step = self.preswitch_signal.make_step(shape_in, shape_out, dt, rng, state)
        postswitch_step = self.postswitch_signal.make_step(shape_in, shape_out, dt, rng, state)

        def step_switchinputs(t):
            return preswitch_step(t) if t < self.switch_time else postswitch_step(t)

        return step_switchinputs


class ConditionalProbe:
    def __init__(self, obj, attr, probe_from):
        if isinstance(obj, nengo.Ensemble):
            self.size_out = obj.dimensions
        if isinstance(obj, nengo.Node):
            self.size_out = obj.size_out
        if isinstance(obj, nengo.Connection):
            self.size_out = obj.size_out

        self.attr = attr
        self.time = probe_from
        self.probed_data = [[] for _ in range(self.size_out)]

    def __call__(self, t, x):
        if x.shape != (self.size_out,):
            raise RuntimeError(
                "Expected dimensions=%d; got shape: %s"
                % (self.size_out, x.shape)
            )
        if t > 0 and t > self.time:
            for i, k in enumerate(x):
                self.probed_data[i].append(k)

    @classmethod
    def setup(cls, obj, attr=None, probe_from=0):
        cond_probe = ConditionalProbe(obj, attr, probe_from)
        output = nengo.Node(cond_probe, size_in=cond_probe.size_out)
        nengo.Connection(obj, output, synapse=0.01)

        return cond_probe

    def get_conditional_probe(self):
        return np.array(self.probed_data).T


class Plotter():
    def __init__(self, trange, rows, cols, dimensions, learning_time, sampling, plot_size=(12, 8), dpi=80, dt=0.001,
                 pre_alpha=0.3):
        self.time_vector = trange
        self.plot_sizes = plot_size
        self.dpi = dpi
        self.n_rows = rows
        self.n_cols = cols
        self.n_dims = dimensions
        self.learning_time = learning_time
        self.sampling = sampling
        self.dt = dt
        self.pre_alpha = pre_alpha

    def plot_testing(self, pre, post, smooth=False):
        fig, axes = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=False)
        fig.set_size_inches(self.plot_sizes)

        learning_time = int((self.learning_time / self.dt) / (self.sampling / self.dt))
        time = self.time_vector[learning_time:, ...]
        pre = pre[learning_time:, ...]
        post = post[learning_time:, ...]

        axes[0, 0].xaxis.set_tick_params(labelsize='xx-large')
        axes[0, 0].yaxis.set_tick_params(labelsize='xx-large')
        axes[0, 0].set_ylim(-1, 1)

        if smooth:
            from scipy.signal import savgol_filter

            pre = np.apply_along_axis(savgol_filter, 0, pre, window_length=51, polyorder=3)
            post = np.apply_along_axis(savgol_filter, 0, post, window_length=51, polyorder=3)

        axes[0, 0].plot(
            time,
            pre,
            # linestyle=":",
            alpha=self.pre_alpha,
            label='Pre')
        axes[0, 0].set_prop_cycle(None)
        axes[0, 0].plot(
            time,
            post,
            label='Post')
        # if self.n_dims <= 3:
        #     axes[ 0, 0 ].legend(
        #             [ f"Pre dim {i}" for i in range( self.n_dims ) ] +
        #             [ f"Post dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        # axes[ 0, 0 ].set_title( "Pre and post decoded on testing phase", fontsize=16 )

        plt.tight_layout()

        return fig

    def plot_results(self, input, pre, post, error, smooth=False):
        fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, squeeze=False)
        fig.set_size_inches(self.plot_sizes)

        for ax in axes.flatten():
            ax.xaxis.set_tick_params(labelsize='xx-large')
            ax.yaxis.set_tick_params(labelsize='xx-large')

        axes[0, 0].plot(
            self.time_vector,
            input,
            label='Input',
            linewidth=2.0)
        # if self.n_dims <= 3:
        #     axes[ 0, 0 ].legend(
        #             [ f"Input dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        axes[0, 0].set_title("Input signal", fontsize=16)

        if smooth:
            from scipy.signal import savgol_filter

            pre = np.apply_along_axis(savgol_filter, 0, pre, window_length=51, polyorder=3)
            post = np.apply_along_axis(savgol_filter, 0, post, window_length=51, polyorder=3)

        axes[1, 0].plot(
            self.time_vector,
            pre,
            # linestyle=":",
            alpha=self.pre_alpha,
            label='Pre')
        axes[1, 0].set_prop_cycle(None)
        axes[1, 0].plot(
            self.time_vector,
            post,
            label='Post')
        # if self.n_dims <= 3:
        #     axes[ 1, 0 ].legend(
        #             [ f"Pre dim {i}" for i in range( self.n_dims ) ] +
        #             [ f"Post dim {i}" for i in range( self.n_dims ) ],
        #             loc='best' )
        axes[1, 0].set_title("Pre and post decoded", fontsize=16)

        if smooth:
            from scipy.signal import savgol_filter

            error = np.apply_along_axis(savgol_filter, 0, error, window_length=51, polyorder=3)
        axes[2, 0].plot(
            self.time_vector,
            error,
            label='Error')
        if self.n_dims <= 3:
            axes[2, 0].legend(
                [f"Error dim {i}" for i in range(self.n_dims)],
                loc='best')
        axes[2, 0].set_title("Error", fontsize=16)

        for ax in axes:
            ax[0].axvline(x=self.learning_time, c="k")

        fig.get_axes()[0].annotate(f"{self.n_rows} neurons, {self.n_dims} dimensions", (0.5, 0.94),
                                   xycoords='figure fraction', ha='center',
                                   fontsize=20
                                   )
        plt.tight_layout()

        return fig

    def plot_ensemble_spikes(self, name, spikes, decoded):
        fig, ax1 = plt.subplots()
        fig.set_size_inches(self.plot_sizes)
        ax1 = plt.subplot(1, 1, 1)
        rasterplot(self.time_vector, spikes, ax1)
        ax1.axvline(x=self.learning_time, c="k")
        ax2 = plt.twinx()
        ax2.plot(self.time_vector, decoded, c="k", alpha=0.3)
        ax1.set_xlim(0, max(self.time_vector))
        ax1.set_ylabel('Neuron')
        ax1.set_xlabel('Time (s)')
        fig.get_axes()[0].annotate(name + " neural activity", (0.5, 0.94),
                                   xycoords='figure fraction', ha='center',
                                   fontsize=20
                                   )

        return fig

    def plot_values_over_time(self, pos_memr, neg_memr, value="conductance"):
        if value == "conductance":
            tit = "Conductances"
            pos_memr = 1 / pos_memr
            neg_memr = 1 / neg_memr
        if value == "resistance":
            tit = "Resistances"
        fig, axes = plt.subplots(self.n_rows, self.n_cols)
        fig.set_size_inches(self.plot_sizes)
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                pos_cond = pos_memr[:int(self.learning_time / self.dt), i, j]
                neg_cond = neg_memr[:int(self.learning_time / self.dt), i, j]
                axes[i, j].plot(pos_cond, c="r")
                axes[i, j].plot(neg_cond, c="b")
                axes[i, j].set_title(f"{j}->{i}")
                axes[i, j].set_yticklabels([])
                axes[i, j].set_xticklabels([])
                plt.subplots_adjust(hspace=0.7)
        fig.get_axes()[0].annotate(f"{tit} over time", (0.5, 0.94),
                                   xycoords='figure fraction', ha='center',
                                   fontsize=20
                                   )
        # plt.tight_layout()

        return fig

    def plot_weights_over_time(self, pos_memr, neg_memr):
        fig, axes = plt.subplots(self.n_rows, self.n_cols)
        fig.set_size_inches(self.plot_sizes)
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                pos_cond = 1 / pos_memr[:int(self.learning_time / self.dt), i, j]
                neg_cond = 1 / neg_memr[:int(self.learning_time / self.dt), i, j]
                axes[i, j].plot(pos_cond - neg_cond, c="g")
                axes[i, j].set_title(f"{j}->{i}")
                axes[i, j].set_yticklabels([])
                axes[i, j].set_xticklabels([])
                plt.subplots_adjust(hspace=0.7)
        fig.get_axes()[0].annotate("Weights over time", (0.5, 0.94),
                                   xycoords='figure fraction', ha='center',
                                   fontsize=20
                                   )
        # plt.tight_layout()

        return fig

    def plot_weight_matrices_over_time(self, weights, n_cols=5, sample_every=0.001):
        n_rows = int(self.learning_time / n_cols) + 1
        fig, axes = plt.subplots(n_rows, n_cols)
        fig.set_size_inches(self.plot_sizes)

        for t, ax in enumerate(axes.flatten()):
            if t <= self.learning_time:
                ax.matshow(weights[int((t / self.dt) / (sample_every / self.dt)), ...],
                           cmap=plt.cm.Blues)
                ax.set_title(f"{t}")
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                plt.subplots_adjust(hspace=0.7)
            else:
                ax.set_axis_off()
        fig.get_axes()[0].annotate("Weights over time", (0.5, 0.94),
                                   xycoords='figure fraction', ha='center',
                                   fontsize=18
                                   )
        # plt.tight_layout()

        return fig


def make_timestamped_dir(root=None):
    if root is None:
        root = "../data/"

    os.makedirs(os.path.dirname(root), exist_ok=True)

    time_string = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    dir_name = root + time_string + "/"
    if os.path.isdir(dir_name):
        raise FileExistsError("The directory already exists")
    dir_images = dir_name + "images/"
    dir_data = dir_name + "data/"
    os.mkdir(dir_name)
    os.mkdir(dir_images)
    os.mkdir(dir_data)

    return dir_name, dir_images, dir_data


def mse_to_rho_ratio(mse, rho):
    return [i for i in np.array(rho) / mse]


def correlations(X, Y):
    import scipy

    pearson_correlations = []
    spearman_correlations = []
    kendall_correlations = []
    for x, y in zip(X.T, Y.T):
        pearson_correlations.append(scipy.stats.pearsonr(x, y)[0])
        spearman_correlations.append(scipy.stats.spearmanr(x, y)[0])
        kendall_correlations.append(scipy.stats.kendalltau(x, y)[0])

    return pearson_correlations, spearman_correlations, kendall_correlations


def gini(array):
    """Calculate the Gini coefficient of exponent numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def save_weights(path, probe):
    np.save(path + "weights.npy", probe[-1].T)


def save_memristors_to_csv(dir, pos_memr, neg_memr):
    num_post = pos_memr.shape[0]
    num_pre = pos_memr.shape[1]

    pos_memr = pos_memr.reshape((pos_memr.shape[0], -1))
    neg_memr = neg_memr.reshape((neg_memr.shape[0], -1))

    header = []
    for i in range(num_post):
        for j in range(num_pre):
            header.append(f"{j}->{i}")
    header = ','.join(header)

    np.savetxt(dir + "pos_resistances.csv", pos_memr, delimiter=",", header=header, comments="")
    np.savetxt(dir + "neg_resistances.csv", neg_memr, delimiter=",", header=header, comments="")
    np.savetxt(dir + "weights.csv", 1 / pos_memr - 1 / neg_memr, delimiter=",", header=header, comments="")


def save_results_to_csv(dir, input, pre, post, error):
    header = []
    header.append(",".join(["input" + str(i) for i in range(input.shape[1])]))
    header.append(",".join(["pre" + str(i) for i in range(pre.shape[1])]))
    header.append(",".join(["post" + str(i) for i in range(post.shape[1])]))
    header.append(",".join(["error" + str(i) for i in range(error.shape[1])]))
    header = ",".join(header)

    with open(dir + "results.csv", "w") as f:
        np.savetxt(f, np.hstack((input, pre, post, error)), delimiter=",", header=header, comments="")


def nested_dict(n, type):
    from collections import defaultdict

    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n - 1, type))

def allow_gpu_growth():
    # Makes the program not crash somehow
    allow_growth = True
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        # print(e)
        pass
