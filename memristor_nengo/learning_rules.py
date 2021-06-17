import warnings

import numpy as np

from nengo.builder import Operator
from nengo.builder.learning_rules import build_or_passthrough, get_post_ens, get_pre_ens
from nengo.learning_rules import LearningRuleType
from nengo.params import Default, NumberParam
from nengo.synapses import Lowpass, SynapseParam

from scipy.stats import truncnorm

# Globals
strategies = ["synapse", "bridge"]
strategy = strategies[0]

learning_rules = ["Hebbian", "Oja", "GHA"]
learning_rule = learning_rules[1]

linearize = True


def get_truncated_normal(mean, sd, low, upp, out_size, in_size):
    try:
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd) \
            .rvs(out_size * in_size) \
            .reshape((out_size, in_size))
    except ZeroDivisionError:
        return np.full((out_size, in_size), mean)


def res2cond(R, r_min, r_max):
    g_min = 1.0 / r_max
    g_max = 1.0 / r_min
    g_curr = 1.0 / R
    g_norm = (g_curr - g_min) / (g_max - g_min)

    return g_norm


# Applies the given voltage to a set of memristors
def apply_voltage(V, memristors, r_max, r_min, exp_a, exp_b):
    forw_exp = exp_a + exp_b * np.abs(V)
    forw_n = np.power((memristors[V > 0] - r_min[V > 0]) / r_max[V > 0], 1 / forw_exp[V > 0])
    memristors[V > 0] = r_min[V > 0] + r_max[V > 0] * np.power(forw_n + 1, forw_exp[V > 0])


# Returns values for the max/min R, and exponent component a and b
def initialise_memristor_attrs(rule, in_size, out_size):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(rule.seed)
        r_min_noisy = get_truncated_normal(rule.r_min, rule.r_min * rule.noise_percentage[0],
                                           0, np.inf, out_size, in_size)
        np.random.seed(rule.seed)
        r_max_noisy = get_truncated_normal(rule.r_max, rule.r_max * rule.noise_percentage[1],
                                           np.max(r_min_noisy), np.inf, out_size, in_size)

    np.random.seed(rule.seed)
    exp_a_noisy = np.random.normal(rule.exp_a, np.abs(rule.exp_a) * rule.noise_percentage[2], (out_size, in_size))
    np.random.seed(rule.seed)
    exp_b_noisy = np.random.normal(rule.exp_b, np.abs(rule.exp_b) * rule.noise_percentage[2], (out_size, in_size))

    return r_min_noisy, r_max_noisy, exp_a_noisy, exp_b_noisy


# Gives the actual signal for a set of memristors
def initialize_memristor_set(rule, in_size, out_size, name):
    np.random.seed(rule.seed + 1) if rule.seed else np.random.seed(rule.seed)
    mem_initial = np.random.normal(1e8, 1e8 * rule.noise_percentage[3], (out_size, in_size))
    memristors = Signal(shape=(out_size, in_size), name=f"{rule}:{name}",
                            initial_value=mem_initial)

    return memristors


# Clips memristors between max and min R when it has been updated
def clip_memristor_values(V, memristors, r_max, r_min):
    memristors[:] = np.where(memristors[:] > r_max[:], r_max[:], memristors[:])
    memristors[:] = np.where(memristors[:] < r_min[:], r_min[:], memristors[:])


def linearize_voltages(volt, V, exp_a, exp_b, r_min, r_max, exp_res, mems):
    # Calculate desired resistance after pulse
    exp_res[volt > 0] = exp_res[volt > 0] - 5e3

    # Set to lowest voltage that goes under desired voltage
    # for cur_volt in [50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.0]:
    #     forw_exp = exp_a + exp_b * np.abs(V) * cur_volt
    #     forw_n = np.power((mems[:] - r_min[:]) / r_max[:], 1 / forw_exp[:])
    #     calc_res = r_min[:] + r_max[:] * np.power(forw_n + 1, forw_exp[:])
    #     mask = np.logical_and(calc_res < exp_res, volt > 0)
    #     volt[mask] = (np.abs(V) * cur_volt)[mask]

    # See if voltage of 0 also
    forw_exp = exp_a + exp_b * np.abs(V)
    forw_n = np.power((mems[:] - r_min[:]) / r_max[:], 1 / forw_exp[:])
    calc_res = r_min[:] + r_max[:] * np.power(forw_n + 1, forw_exp[:])
    # Set them to 0 if they are already below the wanted resistance
    volt[calc_res < exp_res] = 0.0


# applies voltage according to paper from Thomas
def update_memristors_synapse(V, pos_memristors, neg_memristors, r_max, r_min, exp_a, exp_b, exp_res_pos, exp_res_neg):
    V *= 1e-1
    # Check where voltage pulses have to be applied
    pos_volt = np.where(V < np.full_like(V, 0.0), np.full_like(V, 0.0), V)
    neg_volt = np.where(V[:] > np.full_like(V, 0.0)[:], np.full_like(V, 0.0)[:], V[:])
    neg_volt[:] = np.where(neg_volt[:] < np.full_like(V, 0.0)[:], -neg_volt[:], neg_volt[:])

    # Set voltage such that it gives a linear descent
    if linearize:
        linearize_voltages(pos_volt, V, exp_a, exp_b, r_min, r_max, exp_res_pos, pos_memristors)
        linearize_voltages(neg_volt, V, exp_a, exp_b, r_min, r_max, exp_res_neg, neg_memristors)

    # Apply the voltages
    apply_voltage(pos_volt, pos_memristors, r_max, r_min, exp_a, exp_b)
    apply_voltage(neg_volt, neg_memristors, r_max, r_min, exp_a, exp_b)


def update_memristors_bridge(V, m1, m2, m3, m4, r_max, r_min, exp_a, exp_b):
    # Calculate voltage for every memristor
    V_M1 = 1.0/m1 * (1.0/m1 + 1.0/m2) * V
    V_M2 = 1.0/m2 * (1.0/m1 + 1.0/m2) * V
    V_M3 = 1.0/m3 * (1.0/m3 + 1.0/m4) * V
    V_M4 = 1.0/m4 * (1.0/m3 + 1.0/m4) * V

    # Apply the voltage to correct memristors
    apply_voltage(V_M1, m1, r_max, r_min, exp_a, exp_b)
    apply_voltage(V_M2, m2, r_max, r_min, exp_a, exp_b)
    apply_voltage(V_M3, m3, r_max, r_min, exp_a, exp_b)
    apply_voltage(V_M4, m4, r_max, r_min, exp_a, exp_b)


# Calculates the weights (difference between resistances)
def get_weights_synapse(weights, pos_memristors, neg_memristors, r_max, r_min, gain):
    weights[:] = gain * ((1 / pos_memristors[:]) - (1 / neg_memristors[:]))
    # weights[:] = gain * (res2cond(pos_memristors[:], r_min[:], r_max[:]) - res2cond(neg_memristors[:], r_min[:], r_max[:]))

def get_weights_bridge(weights, m1, m2, m3, m4, r_max, r_min, gain):
    weights[:] = (m2 * (m1 + m2)) - (m4 * (m3 + m4))


def find_spikes(input_activities, shape, output_activities=None, invert=False):
    output_size = shape[0]
    input_size = shape[1]

    spiked_pre = np.tile(np.array(np.rint(input_activities), dtype=bool), (output_size, 1))

    if output_activities is not None:
        spiked_post = np.tile(np.expand_dims(np.array(np.rint(output_activities), dtype=bool), axis=1), (1, input_size))
    else:
        spiked_post = np.ones((1, input_size))

    out = np.logical_and(spiked_pre, spiked_post)

    if invert:
        return np.logical_not(out)
    else:
        return out


class mOja(LearningRuleType):
    modifies = "weights"
    probeable = ("error", "activities", "delta", "mems1", "mems2", "mems3", "mems4", "mems5")

    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    post_synapse = SynapseParam("post_synapse", default=None, readonly=True)
    beta = NumberParam("beta", low=0, readonly=True, default=1.0)
    exp_res1 = NumberParam("exp_res1", readonly=True, default=2.3e8)
    exp_res2 = NumberParam("exp_res1", readonly=True, default=2.3e8)
    exp_res3 = NumberParam("exp_res1", readonly=True, default=2.3e8)
    exp_res4 = NumberParam("exp_res1", readonly=True, default=2.3e8)
    exp_res5 = NumberParam("exp_res1", readonly=True, default=2.3e8)
    r_max = NumberParam("r_max", readonly=True, default=2.3e8)
    r_min = NumberParam("r_min", readonly=True, default=200)
    exp_a = NumberParam("exp_a", readonly=True, default=-0.093)
    exp_b = NumberParam("exp_b", readonly=True, default=-0.53)
    gain = NumberParam("gain", readonly=True, default=1e3)

    def __init__(self,
                 pre_synapse=Default,
                 post_synapse=Default,
                 beta=Default,
                 exp_res1=Default,
                 exp_res2=Default,
                 exp_res3=Default,
                 exp_res4=Default,
                 exp_res5=Default,
                 r_max=Default,
                 r_min=Default,
                 exp_a=Default,
                 exp_b=Default,
                 noisy=False,
                 gain=Default,
                 seed=None):

        super().__init__(size_in="post_state")

        self.pre_synapse = pre_synapse
        self.post_synapse = post_synapse
        self.beta = beta
        self.exp_res1 = exp_res1
        self.exp_res2 = exp_res2
        self.exp_res3 = exp_res3
        self.exp_res4 = exp_res4
        self.exp_res5 = exp_res5
        self.r_max = r_max
        self.r_min = r_min
        self.exp_a = exp_a
        self.exp_b = exp_b
        if not noisy:
            self.noise_percentage = np.zeros(4)
        elif isinstance(noisy, float):
            self.noise_percentage = np.full(4, noisy)
        elif isinstance(noisy, list) and len(noisy) == 4:
            self.noise_percentage = noisy
        else:
            raise ValueError(f"Noisy parameter must be int or list of length 4, not {type(noisy)}")
        self.gain = gain
        self.seed = seed

    @property
    def _argdefaults(self):
        return (
            ("pre_synapse", mOja.pre_synapse.default),
            ("post_synapse", mOja.post_synapse.default),
            ("beta", mOja.beta.default),
            ("exp_res1", mOja.exp_res1.default),
            ("exp_res2", mOja.exp_res2.default),
            ("exp_res3", mOja.exp_res3.default),
            ("exp_res4", mOja.exp_res4.default),
            ("exp_res5", mOja.exp_res5.default),
            ("r_max", mOja.r_max.default),
            ("r_min", mOja.r_min.default),
            ("exp_a", mOja.exp_a.default),
            ("exp_b", mOja.exp_b.default),
        )

class SimmOja(Operator):
    def __init__(
            self,
            pre_filtered,
            post_filtered,
            beta,
            mems1,
            mems2,
            mems3,
            mems4,
            mems5,
            weights,
            noise_percentage,
            gain,
            exp_res1,
            exp_res2,
            exp_res3,
            exp_res4,
            exp_res5,
            r_min,
            r_max,
            exp_a,
            exp_b,
            states=None,
            tag=None
    ):
        super(SimmOja, self).__init__(tag=tag)

        # scale beta by gain for it to have the expected effect
        self.beta = beta * gain / 2
        self.noise_percentage = noise_percentage
        self.gain = gain
        self.exp_res1 = exp_res1
        self.exp_res2 = exp_res2
        self.exp_res3 = exp_res3
        self.exp_res4 = exp_res4
        self.exp_res5 = exp_res5
        self.r_min = r_min
        self.r_max = r_max
        self.exp_a = exp_a
        self.exp_b = exp_b

        self.sets = [] + ([] if states is None else [states])
        self.incs = []
        self.reads = [pre_filtered, post_filtered]
        self.updates = [weights, mems1, mems2, mems3, mems4, mems5]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def weights(self):
        return self.updates[0]

    @property
    def mems1(self):
        return self.updates[1]

    @property
    def mems2(self):
        return self.updates[2]

    @property
    def mems3(self):
        return self.updates[3]

    @property
    def mems4(self):
        return self.updates[4]

    @property
    def mems5(self):
        return self.updates[5]

    def _descstr(self):
        return "pre=%s, post=%s -> %s" % (self.pre_filtered, self.post_filtered, self.weights)

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]

        mems1 = signals[self.mems1]
        mems2 = signals[self.mems2]
        mems3 = signals[self.mems3]
        mems4 = signals[self.mems4]
        mems5 = signals[self.mems5]
        weights = signals[self.weights]

        beta = self.beta
        gain = self.gain
        exp_res1 = self.exp_res1
        exp_res2 = self.exp_res2
        exp_res3 = self.exp_res3
        exp_res4 = self.exp_res4
        exp_res5 = self.exp_res5
        r_min = self.r_min
        r_max = self.r_max
        exp_a = self.exp_a
        exp_b = self.exp_b

        # overwrite initial transform with memristor-based weights
        if strategy == "synapse":
            get_weights_synapse(weights, mems1, mems2, r_max, r_min, gain)
        if strategy == "bridge":
            get_weights_bridge(weights, mems1, mems2, mems3, mems4, r_max, r_min, gain)

        def step_simmoja():
            if learning_rule == "Hebbian":
                weights_delta = np.outer(post_filtered, pre_filtered)
            # calculate the magnitude of the update based on Hebbian learning rule
            elif learning_rule == "Oja":
                # calculate the magnitude of the update based on Oja learning rule
                post_squared = post_filtered * post_filtered
                forgetting = beta * weights * post_squared
                hebbian = np.outer(post_filtered, pre_filtered)
                weights_delta = hebbian - forgetting
            else:
                # Calculate the magnitude of the update based on GHA learning rule
                pos_comp = np.dot(post_filtered, pre_filtered.T)
                neg_comp = beta * np.tril(np.dot(np.dot(post_filtered, post_filtered.T), weights))
                weights_delta = pos_comp - neg_comp


            # some memristors are adjusted erroneously if we don't filter
            spiked_map = find_spikes(pre_filtered, weights.shape, post_filtered, invert=True)
            weights_delta[spiked_map] = 0

            # set update direction and magnitude
            V = np.sign(weights_delta)

            if strategy == "synapse":
                # clip values outside [R_0,R_1]
                clip_memristor_values(V, mems1, r_max, r_min)
                clip_memristor_values(V, mems2, r_max, r_min)
                # update the two memristor pairs
                update_memristors_synapse(V, mems1, mems2, r_max, r_min, exp_a, exp_b, exp_res1, exp_res2)
                # update network weights
                get_weights_synapse(weights, mems1, mems2, r_max, r_min, gain)

            if strategy == "bridge":
                # clip values outside [R_0,R_1]
                clip_memristor_values(V, mems1, r_max, r_min)
                clip_memristor_values(V, mems2, r_max, r_min)
                clip_memristor_values(V, mems3, r_max, r_min)
                clip_memristor_values(V, mems4, r_max, r_min)
                # update the 4 memristors
                update_memristors_bridge(V, mems1, mems2, mems3, mems4, r_max, r_min, exp_a, exp_b)
                # update network weights
                get_weights_bridge(weights, mems1, mems2, mems3, mems4, r_max, r_min, gain)

        return step_simmoja


################ NENGO DL #####################
import nengo_dl
import tensorflow as tf
from nengo.builder import Signal
from nengo.builder.operator import Reset, DotInc, Copy

import nengo_dl
from nengo_dl.builder import Builder, OpBuilder, NengoBuilder
from nengo.builder import Builder as NengoCoreBuilder

@NengoCoreBuilder.register(mOja)
def build_moja(model, moja, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_filtered = build_or_passthrough(model, moja.pre_synapse, pre_activities)
    post_filtered = build_or_passthrough(model, moja.post_synapse, post_activities)

    r_min_noisy, r_max_noisy, exp_a_noisy, exp_b_noisy = \
        initialise_memristor_attrs(moja, pre_filtered.shape[0], post_filtered.shape[0])

    exp_res1 = np.full((pre_filtered.shape[0], post_filtered.shape[0]), 1e8)
    exp_res2 = np.full((pre_filtered.shape[0], post_filtered.shape[0]), 1e8)
    exp_res3 = np.full((pre_filtered.shape[0], post_filtered.shape[0]), 1e8)
    exp_res4 = np.full((pre_filtered.shape[0], post_filtered.shape[0]), 1e8)
    exp_res5 = np.full((pre_filtered.shape[0], post_filtered.shape[0]), 1e8)

    mems1 = initialize_memristor_set(moja, pre_filtered.shape[0], post_filtered.shape[0], "mems1")
    mems2 = initialize_memristor_set(moja, pre_filtered.shape[0], post_filtered.shape[0], "mems2")
    mems3 = initialize_memristor_set(moja, pre_filtered.shape[0], post_filtered.shape[0], "mems3")
    mems4 = initialize_memristor_set(moja, pre_filtered.shape[0], post_filtered.shape[0], "mems4")
    mems5 = initialize_memristor_set(moja, pre_filtered.shape[0], post_filtered.shape[0], "mems5")

    model.sig[conn]["mems1"] = mems1
    model.sig[conn]["mems2"] = mems2
    model.sig[conn]["mems3"] = mems3
    model.sig[conn]["mems4"] = mems4
    model.sig[conn]["mems5"] = mems5

    model.add_op(
        SimmOja(
            pre_filtered,
            post_filtered,
            moja.beta,
            model.sig[conn]["mems1"],
            model.sig[conn]["mems2"],
            model.sig[conn]["mems3"],
            model.sig[conn]["mems4"],
            model.sig[conn]["mems5"],
            model.sig[conn]["weights"],
            moja.noise_percentage,
            moja.gain,
            exp_res1,
            exp_res2,
            exp_res3,
            exp_res4,
            exp_res5,
            r_min_noisy,
            r_max_noisy,
            exp_a_noisy,
            exp_b_noisy
        )
    )

    # expose these for probes
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
    model.sig[rule]["mems1"] = mems1
    model.sig[rule]["mems2"] = mems2
    model.sig[rule]["mems3"] = mems3
    model.sig[rule]["mems4"] = mems4
    model.sig[rule]["mems5"] = mems5
