from hopfieldNetwork import hopfieldNet
from solverFile import solverClass
import numpy as np
import copy


def to_triangular(matrix):
    return matrix[np.triu_indices_from(matrix, 1)]


def from_triangular(size, arr, diagonal_value):
    matrix = np.zeros([size, size])
    matrix[np.triu_indices_from(matrix, 1)] = arr
    matrix += matrix.T
    np.fill_diagonal(matrix, diagonal_value)
    return matrix


def dice_coefficient(p1, p2):
    p = 2 * np.sum(np.floor(0.6*(p1 + p2)))
    n = np.sum(p1)+np.sum(p2)
    return p/n


def evaluate_stability(network, patterns):
    overall_error = 0
    for p in patterns:
        network.present_pattern(p)
        network.step(eval_epochs)
        output = netFisher.s
        overall_error += dice_coefficient(p, output)
    return overall_error/len(patterns)


ETA = 0.0001  # learning rate
NTRAIN = 2000  # number of epochs
SPARSITY = 0.1  # number of zeros: SPARSITY = 0.1 means 10% ones and 90% zeros
IMAGE_SIZE = 10  # the size of our pattern will be (IMAGE_SIZE x IMAGE_SIZE)
eval_f = 1  # evaluation frequency (every eval_f-th iteration)
TRIALS = 200  # number of trials over which the results will be averaged
less_changed_weight_value = 0.00
# the learning rate of weights which are considered important have a
# learning rate of ETA * less_changed_weight_value
n_stored_patterns = 5  # number of patterns that are stored before learning
n_new_patterns = 1  # number of patterns to be learned
assert n_new_patterns == 1, "Not implemented"
n_tot_patterns = n_stored_patterns + n_new_patterns  # n of patterns created
number_of_changed_values = 4750
# the number of weigths that are changed is 2*number_of_changed_values
# (The factor of 2 is because of the symmetry of the weight matrix)
eval_epochs = 10  # how many steps are run before dice coefficient computed

# whether to run experiments with different learning rules
RUN_FL = False  # similar to FLT, I prefer the local version
RUN_FIH = False  # the experiment shows it's the same...
RUN_FLT = True

# Define variables
Errors = {}
Errors['trad_80_N'] = -np.ones(shape=(TRIALS, NTRAIN))
Errors['trad_80_O'] = -np.ones(shape=(TRIALS, NTRAIN))
if RUN_FL:
    Errors['FL_N'] = -np.ones(shape=(TRIALS, NTRAIN))
    Errors['FL_O'] = -np.ones(shape=(TRIALS, NTRAIN))
if RUN_FLT:
    Errors['FLT_N'] = -np.ones(shape=(TRIALS, NTRAIN))
    Errors['FLT_O'] = -np.ones(shape=(TRIALS, NTRAIN))
Errors['FI_N'] = -np.ones(shape=(TRIALS, NTRAIN))
Errors['FI_O'] = -np.ones(shape=(TRIALS, NTRAIN))
if RUN_FIH:
    Errors['FIH_N'] = -np.ones(shape=(TRIALS, NTRAIN))
    Errors['FIH_O'] = -np.ones(shape=(TRIALS, NTRAIN))


print('**Started Learning**')

for trial in range(0, TRIALS):
    if trial % eval_f == 0:
        print('Running trial ', trial+1, ' / ', TRIALS)

    # preparing patterns
    solver = solverClass()
    patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, n_tot_patterns)
    netFisher = hopfieldNet(IMAGE_SIZE, ETA, SPARSITY)
    original_patterns = copy.deepcopy(patterns)
    patterns = patterns - SPARSITY

    # learning patterns
    p = np.zeros(shape=(IMAGE_SIZE**2, IMAGE_SIZE**2))
    for i in range(n_stored_patterns):
        p += np.outer(patterns[:, i], patterns[:, i])
        netFisher.append_pattern(patterns[:, i], NTRAIN)
    w1 = p/70  # TODO why 70?


# ========== H ========== #
# Traditional learning rule
    wF = w1
    for epoch in range(NTRAIN):
        diminish_lr = 2 * number_of_changed_values / (IMAGE_SIZE**2)**2
        z = diminish_lr * ETA * (np.outer(patterns[:, n_stored_patterns],
                                 patterns[:, n_stored_patterns]) - wF)

        # training
        perturbation_vector = z
        wF = wF + perturbation_vector
        netFisher.set_weights(wF)

        # checking stability of patterns after eval_epochs iterations
        Errors['trad_80_O'][trial, epoch] = evaluate_stability(  # old patterns
            netFisher, original_patterns.T[:n_stored_patterns])
        Errors['trad_80_N'][trial, epoch] = evaluate_stability(  # new pattern
            netFisher, [original_patterns.T[n_stored_patterns]])

    wH_final = netFisher.w

# ========= FL ========= #
# Now perturbing the weights using the value of the weight (quantile)
    if RUN_FL:
        wF = w1
        for epoch in range(NTRAIN):
            z = ETA * (np.outer(patterns[:, n_stored_patterns],
                                patterns[:, n_stored_patterns]) - wF)
            netFisher.curvature = np.abs(w1)  # not an actual curvature
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))

            copied_curvature_tri = to_triangular(netFisher.curvature)
            weight_perturbation_tri = to_triangular(weight_perturbation)
            small_idx = np.argsort(copied_curvature_tri, axis=None)
            weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
            weight_perturbation = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)

            if (epoch % 100) == 0:
                print('The number of perturbed weights is (case 1):   ',
                      np.sum(weight_perturbation))

            # training
            xyz = np.zeros(np.shape(w1))
            xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            Errors['FL_O'][trial, epoch] = evaluate_stability(  # old patterns
                netFisher, original_patterns.T[:n_stored_patterns])
            Errors['FL_N'][trial, epoch] = evaluate_stability(  # new pattern
                netFisher, [original_patterns.T[n_stored_patterns]])

        wFL_final = netFisher.w

# ======== FLT ========= #
# Now perturbing the weights using the value of the weight (threshold)
    if RUN_FLT:
        wF = w1
        for epoch in range(NTRAIN):
            z = ETA * (np.outer(patterns[:, n_stored_patterns],
                                patterns[:, n_stored_patterns]) - wF)
            netFisher.curvature = np.abs(w1)
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
            np.fill_diagonal(weight_perturbation, 1)

            weight_perturbation[np.abs(w1) < 0.008] = 1
            np.fill_diagonal(weight_perturbation, 1)
            if (epoch % 100) == 0:
                print('The number of perturbed weights is (case 2):   ',
                      np.sum(weight_perturbation))

            # training
            xyz = np.zeros(np.shape(w1))
            xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            Errors['FLT_O'][trial, epoch] = evaluate_stability(  # old patterns
                netFisher, original_patterns.T[:n_stored_patterns])
            Errors['FLT_N'][trial, epoch] = evaluate_stability(  # new pattern
                netFisher, [original_patterns.T[n_stored_patterns]])

        # wFLT_final = netFisher.w


# ======== FI ======== #
# Weights perturbed using Fisher Information
    wF = w1
    for epoch in range(NTRAIN):
        z = ETA * (np.outer(patterns[:, n_stored_patterns],
                            patterns[:, n_stored_patterns]) - wF)
        netFisher.calculate_fisher_information(patterns[:, 0:n_stored_patterns])
        weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))

        copied_curvature_tri = to_triangular(netFisher.curvature)
        weight_perturbation_tri = to_triangular(weight_perturbation)
        small_idx = np.argsort(copied_curvature_tri, axis=None)
        weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
        weight_perturbation = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)

        # training
        xyz = np.zeros(np.shape(w1))
        xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
        perturbation_vector = weight_perturbation * z
        wF = wF + perturbation_vector
        netFisher.set_weights(wF)

        # checking stability of patterns after eval_epochs iterations
        Errors['FI_O'][trial, epoch] = evaluate_stability(  # old patterns
            netFisher, original_patterns.T[:n_stored_patterns])
        Errors['FI_N'][trial, epoch] = evaluate_stability(  # new pattern
            netFisher, [original_patterns.T[n_stored_patterns]])


# ======== FIH ========= #
# Now perturbing the weights using Hebbian way for Fisher information
    if RUN_FIH:
        wF = w1
        for epoch in range(NTRAIN):
            z = ETA * (np.outer(patterns[:, n_stored_patterns],
                                patterns[:, n_stored_patterns]) - wF)
            netFisher.calculate_fisher_information_hebbian(patterns[:, 0:n_stored_patterns])
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))

            copied_curvature_tri = to_triangular(netFisher.curvature)
            weight_perturbation_tri = to_triangular(weight_perturbation)
            small_idx = np.argsort(copied_curvature_tri, axis=None)
            weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
            weight_perturbation = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)

            # training
            xyz = np.zeros(np.shape(w1))
            xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            Errors['FIH_O'][trial, epoch] = evaluate_stability(  # old patterns
                netFisher, original_patterns.T[:n_stored_patterns])
            Errors['FIH_N'][trial, epoch] = evaluate_stability(  # new pattern
                netFisher, [original_patterns.T[n_stored_patterns]])

        # wFIH_final = netFisher.w


filename = "../Complete_errors_stored{}_size{}_spars{}_etasmall.npz".format(
    n_stored_patterns, IMAGE_SIZE, SPARSITY)
np.savez(filename, **Errors)

print('**Finished**')
