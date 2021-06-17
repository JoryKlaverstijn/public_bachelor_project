from main_experiment.nengo_HNN import *
from mnist.mnist import converge, get_random_images, add_img_noise, show_digit_images
from tensorflow import keras
from memristor_nengo.extras import allow_gpu_growth
from memristor_continuous.formulas import *


def single_it(seed, beta, sim_time, dt, noise_amnt, model_type, cont_beta,
              stored_pattern_amnt, stored_pattern_nums, query_pattern_amnt, query_pattern_num, acc_thresh):
    # Makes the FFN work
    allow_gpu_growth()

    # Load processed mnist images and retrieve a subset
    mnist_array = np.load("../mnist/mnist_mOja_array.npy", allow_pickle=True)
    patterns = get_random_images(stored_pattern_amnt, mnist_array, stored_pattern_nums)
    # np.random.shuffle(patterns)

    # Load FFN classifier model
    model = keras.models.load_model("../mnist/mnist_FFN")

    # Choose a query pattern, normalize it, and add noise
    query_patterns = get_random_images(query_pattern_amnt, mnist_array, [query_pattern_num])
    query_patterns_pre = query_patterns.copy()
    for pat in query_patterns:
        add_img_noise(pat, noise_amnt) # Adds noise to every image
        # hide_img_partial(pat, noise_amnt) # Hides bottom part of every image

    converged_patterns = []
    # Run continuous HNN
    if model_type == "cont":
        cont_patterns = []
        for pat in patterns:
            cont_patterns.append((pat + 2.0) / 4.0)
        cont_patterns = np.asarray(cont_patterns)

        for pat in query_patterns:
            q_pat = (pat + 2.0) / 4.0
            converged_patterns.append(update_pattern(q_pat.T, cont_patterns.T, cont_beta))
            np.round(converged_patterns)

    # Run Nengo (m)Oja HNN
    elif model_type == "Oja" or model_type == "mOja":
        # Train HNN
        weights1, pre_syn, trange = learn_weights_nengo(seed, beta, sim_time, patterns, dt, model_type)
        weights = weights1[-1].copy()
        weights = weights + weights.T - np.diag(np.diag(weights))
        # Converge pattern using HNN weights
        for pat in query_patterns:
            converged_pattern = np.dot(weights, pat)
            if model_type == "mOja":
                threshold = np.max(converged_pattern) / 2
            else:
                threshold = 0
            converged_pattern[converged_pattern >threshold] = 1.0
            converged_pattern[converged_pattern < threshold] = 0.0
            # converged_patterns.append(converged_pattern) # dot product convergence
            converged_patterns.append(converge(pat-0.5, weights.copy(), 30)) # sequential update convergence

    # Predict converged image with FFN
    predictions = []
    accuracy = 0.0
    for pat in converged_patterns:
        pred_input = pat.reshape((1, 14, 14, 1)).copy()
        prediction = model.predict(pred_input)
        predictions.append(np.argmax(prediction))
        # print("argmax:", np.argmax(prediction), "query pattern:", query_pattern_num)
        if np.argmax(prediction) == query_pattern_num:
            accuracy += 1.0 / query_pattern_amnt

    # Show the stored images and the converged image
    if accuracy < acc_thresh:
        print("accuracy of", model_type, ":", accuracy)
        print("classifications:", predictions)
        if model_type == "mOja" or model_type == "Oja":
        #    _, s1 = plot_weights_interactive(weights, trange)
            pass
        show_digit_images(converged_patterns, "Converged patterns", predictions) #
        show_digit_images(query_patterns, "Query Patterns")
        # show_digit_images(query_patterns_pre, "Query Patterns")
        show_digit_images(patterns, "Stored patterns")
        plt.show()

    return accuracy













