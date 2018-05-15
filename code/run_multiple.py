from hopfieldNetwork import hopfieldNet
from solverFile import solverClass
import numpy as np
import copy
from scipy.special import expit

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


def evaluate_stability(network, patterns, average=True):
    error = np.empty(len(patterns))
    for i, p in enumerate(patterns):
        network.present_pattern(p)
        network.step(eval_epochs)
        output = netFisher.s
        error[i] = dice_coefficient(p, output)
    if average:
        return np.mean(error)
    return error


ETA = 0.001  # learning rate
SPARSITY = 0.1  # number of zeros: SPARSITY = 0.1 means 10% ones and 90% zeros
IMAGE_SIZE = 10  # the size of our pattern will be (IMAGE_SIZE x IMAGE_SIZE)
eval_f = 1  # evaluation frequency (every eval_f-th iteration)
TRIALS = 10  # number of trials over which the results will be averaged
less_changed_weight_value = 0.00
# the learning rate of weights which are considered important have a
# learning rate of ETA * less_changed_weight_value
epochs_patterns_presented = 20#30
n_stored_patterns = 5
n_new_patterns = 30#20
n_tot_patterns = n_stored_patterns + n_new_patterns  # n of patterns created
NTRAIN = epochs_patterns_presented * n_new_patterns  # number of epochs
number_of_changed_values = 4750
# the number of weigths that are changed is 2*number_of_changed_values
# (The factor of 2 is because of the symmetry of the weight matrix)
eval_epochs = 10  # how many steps are run before dice coefficient computed

# whether to run experiments with different learning rules
RUN_FL = True  # similar to FLT, I prefer the local version
RUN_FI = False  # FI
RUN_FIH = False  # the experiment shows it's the same...
RUN_FLT = False
RUN_SCL = True

# Define variables
Errors = {}
Errors['trad_80_N'] = -np.ones(shape=(TRIALS, NTRAIN, n_tot_patterns))
if RUN_FL:
    Errors['FL_N'] = -np.ones(shape=(TRIALS, NTRAIN, n_tot_patterns))
if RUN_FLT:
    Errors['FLT_N'] = -np.ones(shape=(TRIALS, NTRAIN, n_tot_patterns))
if RUN_FI:
    Errors['FI_N'] = -np.ones(shape=(TRIALS, NTRAIN, n_tot_patterns))
if RUN_FIH:
    Errors['FIH_N'] = -np.ones(shape=(TRIALS, NTRAIN, n_tot_patterns))
if RUN_SCL:
    Errors['FL_SCL'] = -np.ones(shape=(TRIALS, NTRAIN, n_tot_patterns))


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
    p = np.random.random((IMAGE_SIZE**2, IMAGE_SIZE**2))
    for i in range(n_stored_patterns):
        p += np.outer(patterns[:, i], patterns[:, i])
        netFisher.append_pattern(patterns[:, i], NTRAIN)
    w1 = p/70


# ========== H ========== #
# Traditional learning rule
    wF = np.copy(w1)
    for epoch in range(NTRAIN):
        id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
        pattern_taught = patterns[:, id_pattern_taught]

        diminish_lr = 2 * number_of_changed_values / (IMAGE_SIZE**2)**2
        z = diminish_lr * ETA * (np.outer(pattern_taught, pattern_taught) - wF)

        # training
        perturbation_vector = z
        wF = wF + perturbation_vector
        netFisher.set_weights(wF)

        # checking stability of patterns after eval_epochs iterations
        Errors['trad_80_N'][trial, epoch] = evaluate_stability(  # old patterns
            netFisher, original_patterns.T[:n_tot_patterns], average=False)

    wH_final = netFisher.w

# ========= FL ========= #
# Now perturbing the weights using the value of the weight (quantile)
    if RUN_FL:
        wF = np.copy(w1)
        for epoch in range(NTRAIN):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)
            # netFisher.curvature = np.abs(w1)  # not an actual curvature
            netFisher.curvature = np.abs(wF)  # not an actual curvature
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
            xyz[weight_perturbation == 1] = wF[weight_perturbation == 1]
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            Errors['FL_N'][trial, epoch] = evaluate_stability(  # old patterns
                netFisher, original_patterns.T[:n_tot_patterns], average=False)

        wFL_final = netFisher.w

# ======== FLT ========= #
# Now perturbing the weights using the value of the weight (threshold)
    if RUN_FLT:
        wF = np.copy(w1)
        for epoch in range(NTRAIN):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)
            # netFisher.curvature = np.abs(w1)
            netFisher.curvature = np.abs(wF)
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
            np.fill_diagonal(weight_perturbation, 1)

            # weight_perturbation[np.abs(w1) < 0.008] = 1
            weight_perturbation[np.abs(wF) < 0.008] = 1
            np.fill_diagonal(weight_perturbation, 1)
            if (epoch % 100) == 0:
                print('The number of perturbed weights is (case 2):   ',
                      np.sum(weight_perturbation))

            # training
            xyz = np.zeros(np.shape(w1))
            # xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            xyz[weight_perturbation == 1] = wF[weight_perturbation == 1]
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            Errors['FLT_N'][trial, epoch] = evaluate_stability(  # old patterns
                netFisher, original_patterns.T[:n_tot_patterns], average=False)

        # wFLT_final = netFisher.w


# ======== FI ======== #
# Weights perturbed using Fisher Information
    if RUN_FI:
        wF = np.copy(w1)
        for epoch in range(NTRAIN):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)

            netFisher.calculate_fisher_information(  # TODO review
                patterns[:, :id_pattern_taught])
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))

            copied_curvature_tri = to_triangular(netFisher.curvature)
            weight_perturbation_tri = to_triangular(weight_perturbation)
            small_idx = np.argsort(copied_curvature_tri, axis=None)[::-1]
            weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
            weight_perturbation = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)

            # training
            xyz = np.zeros(np.shape(w1))
            xyz[weight_perturbation == 1] = wF[weight_perturbation == 1]
            # xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            Errors['FI_N'][trial, epoch] = evaluate_stability(  # old patterns
                netFisher, original_patterns.T[:n_tot_patterns], average=False)


# ======== FIH ========= #
# Now perturbing the weights using Hebbian way for Fisher information
    if RUN_FIH:
        wF = np.copy(w1)
        for epoch in range(NTRAIN):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)

            netFisher.calculate_fisher_information_hebbian(  # TODO review
                patterns[:, :id_pattern_taught])
            # weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(wF))

            copied_curvature_tri = to_triangular(netFisher.curvature)
            weight_perturbation_tri = to_triangular(weight_perturbation)
            small_idx = np.argsort(copied_curvature_tri, axis=None)
            weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
            weight_perturbation = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)

            # training
            xyz = np.zeros(np.shape(w1))
            # xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            xyz[weight_perturbation == 1] = wF[weight_perturbation == 1]
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            Errors['FIH_N'][trial, epoch] = evaluate_stability(  # new pattern
                netFisher, original_patterns.T[:n_tot_patterns], average=False)

        # wFIH_final = netFisher.w
# ========= SCL ========= #
# Now perturbing the weights proportional to its value
    if RUN_SCL:
        wF = np.copy(w1)
        for epoch in range(NTRAIN):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)
            netFisher.curvature = np.abs(wF)  # not an actual curvature
            weight_perturbation = np.exp(-netFisher.curvature*40)

            if (epoch % 100) == 0:
                print('perturbation (25,50,75% , min, max): ',np.percentile(weight_perturbation,(25,50,75)),np.min(weight_perturbation), np.max(weight_perturbation)  )

            # training
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            Errors['FL_SCL'][trial, epoch] = evaluate_stability(  # old patterns
                netFisher, original_patterns.T[:n_tot_patterns], average=False)


filename = "../Complete_errors_stored{}_size{}_spars{}_ALL.npz".format(
    n_tot_patterns, IMAGE_SIZE, SPARSITY)
np.savez(filename, **Errors)
print(filename)
print('**Finished**')
