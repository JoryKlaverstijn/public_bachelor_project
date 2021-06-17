from main_experiment.single_iteration import single_it
import random
import numpy as np


# Nengo parameters
seed = 42 # Seed for Nengo model
beta = 0.5 # Forgetting component
sim_time = 10.0  # total simulation time (seconds)
dt = 0.005 # precision of nengo simulation

# continuous parameters
cont_beta = 1000000.0  # Temperature parameter

# Experiment parameters
mOja = True  # test mOja model
Oja = True  # test Oja model
continuous = True  # Test continuous HNN
stored_pattern_amnt = 10  # Amount of images per digit stored inside the HNN
stored_diff_digit_amnt = 5  # Amount of different digits stored inside the HNN (1-10)
query_pattern_amnt = 100  # Amount of query patterns converged/tested on each run
run_amnt = 100  # Amount of digit sets used to test the models
noise_amnt = 0.05  # Amount of noise added to the images

# Show images if total accuracy is below the following threshold (set to >1.0 to view all, <0.0 to view none)
acc_thresh = -1.0

# Initialize accuracies
tot_accuracy_mOja = 0.0
tot_accuracy_continuous = 0.0
tot_accuracy_Oja = 0.0

mOja_accurasies = []
cont_accurasies = []
Oja_accurasies = []

for it in range(run_amnt):
    # Take set amount of random digits from 0-9
    stored_digits = []
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for _ in range(stored_diff_digit_amnt):
        rand_nr = random.randint(0, len(digits) - 1)
        stored_digits.append(digits[rand_nr])
        digits.pop(rand_nr)

    # Run mOja HNN on the given digits
    if mOja:
        accuracy_mOja = single_it(seed, beta, sim_time, dt, noise_amnt, "mOja", cont_beta,
                                  stored_pattern_amnt, stored_digits, query_pattern_amnt, stored_digits[0],
                                  acc_thresh)
        tot_accuracy_mOja += accuracy_mOja / run_amnt
        mOja_accurasies.append(accuracy_mOja)

    # Run continuous HNN on the given digits
    if continuous:
        accuracy_continuous = single_it(seed, beta, sim_time, dt, noise_amnt, "cont", cont_beta,
                                        stored_pattern_amnt, stored_digits, query_pattern_amnt, stored_digits[0],
                                        acc_thresh)
        tot_accuracy_continuous += accuracy_continuous / run_amnt
        cont_accurasies.append(accuracy_continuous)

    # Run Oja HNN on the given digits
    if Oja:
        accuracy_Oja = single_it(seed, beta, sim_time, dt, noise_amnt, "Oja", cont_beta,
                                 stored_pattern_amnt, stored_digits, query_pattern_amnt,
                                 stored_digits[0], acc_thresh)
        tot_accuracy_Oja += accuracy_Oja / run_amnt
        Oja_accurasies.append(accuracy_Oja)

    # Print the iteration number and the seperate accuracies
    print("test iteration:", it, "with numbers:", stored_digits)
    if mOja:
        print("\taccuracy mOja:", str(round(accuracy_mOja, 5)))
    if continuous:
        print("\taccuracy continuous:", str(round(accuracy_continuous, 5)))
    if Oja:
        print("\taccuracy Oja:", str(round(accuracy_Oja, 5)))

# Print final accuracies
print("different digits:", stored_diff_digit_amnt)
print("noise:", noise_amnt)
print("total accuracy for mOja HNN:", str(round(tot_accuracy_mOja, 5)), "standard deviation:", np.std(mOja_accurasies))
print("total accuracy for continuous HNN:", str(round(tot_accuracy_continuous, 5)), "standard deviation:", np.std(cont_accurasies))
print("total accuracy for Oja HNN:", str(round(tot_accuracy_Oja, 5)), "standard deviation:", np.std(Oja_accurasies))

