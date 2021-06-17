from memristor_nengo.extras import *
from memristor_nengo.learning_rules import mOja
from nengo.utils.ensemble import tuning_curves
# from mnist import *


def learn_weights_nengo(seed, beta, sim_time, patterns, dt, model_type):
    # The function that returns the correct pattern (same amnt of time per pattern)
    # print("pattern amount:", patterns.shape[0])
    # print("time per pattern:", sim_time / patterns.shape[0])

    # Shows every pattern, every second
    def node_function(t):
        pat_amnt = patterns.shape[0]
        time_per_pat = 1.0 / pat_amnt
        index = int((t % 1) % time_per_pat)
        return patterns[index]

    # Shows every pattern for x seconds at a time
    def second_node_function(t):
        pat_amnt = patterns.shape[0]
        time_per_pat = sim_time / pat_amnt
        index = min(int(t / time_per_pat), pat_amnt - 1)
        return patterns[index]


    # The model
    model = nengo.Network(seed=seed)
    with model:
        inp = nengo.Node(node_function)
        pre = nengo.Ensemble(len(patterns[0]), 1, seed=seed)  # ,neuron_type=AdaptiveLIFLateralInhibition()
        nengo.Connection(inp, pre.neurons, seed=seed)
        conn = nengo.Connection(pre.neurons, pre.neurons, transform=np.zeros((pre.n_neurons, pre.n_neurons)), seed=seed)
        if model_type == "Oja":
            conn.learning_rule_type = nengo.learning_rules.Oja(beta=beta)
        else:
            conn.learning_rule_type = mOja(beta=beta, noisy=[0.01, 0.01, 0.01, 0.001])

        # Generate probes to get information from simulation
        pre_probe = nengo.Probe(pre.neurons)
        decoded_pre_probe = nengo.Probe(pre, "decoded_output")
        weight_probe = nengo.Probe(conn, "weights")
        input_probe = nengo.Probe(inp)
        if isinstance(conn.learning_rule_type, mOja):
            mems1_probe = nengo.Probe(conn.learning_rule, "mems1")
            mems2_probe = nengo.Probe(conn.learning_rule, "mems2")


    # Run simulation for sim_time seconds
    with nengo.Simulator(model, seed=seed, dt=dt) as sim:
        sim.run(sim_time)
        eval_points, activities = tuning_curves(pre, sim)

    # Other plots
    # _, s1 = plot_neural_frequency(pre_syn, trange)
    # _, s2 = plot_weights_video(sim.data[weight_probe], sim.trange())
    # plot_memristors(sim.data[mems1_probe][:,90:92,90:92], sim.data[mems2_probe][:,90:92,90:92], sim.trange())
    # plt.show()
    # plot_neural_spiking(pre_syn, trange)

    if model_type == "mOja":
        return sim.data[weight_probe], sim.data[pre_probe], sim.trange()

    # return sim.data[weight_probe], sim.data[pre_probe], sim.data[mems1_probe], sim.data[mems2_probe], sim.trange()
    return sim.data[weight_probe], sim.data[pre_probe], sim.trange()





