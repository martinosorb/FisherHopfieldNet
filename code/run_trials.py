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

ETA = 0.001  # learning rate
NTRAIN = 2000  # number of epochs
NUM_PATTERNS = 1000  # number of patterns created
SPARSITY = 0.1  # number of zeros: e.g. SPARSITY = 0.1 means 10% ones and 90% zeros
IMAGE_SIZE = 10  # the size of our created pattern will be (IMAGE_SIZE x IMAGE_SIZE)
eval_f = 1  # evaluation frequency (every eval_f-th iteration) NOTE: currently not implemented
TRIALS = 300  # number of trials over which the results will be averaged in order to get smooth results
less_changed_weight_value = 0.00
# the learning rate of weights which are considered important have a
# learning rate of ETA * less_changed_weight_value
stored_patterns = 5  # number of patterns that are stored in the network before learning the new pattern
number_of_changed_values = 4750
# the number of weigths that are changed is 2*number_of_changed_values
# (The factor of 2 is because of the symmetry of the weight matrix)
eval_epochs = 100  # how many steps are run before dice coefficient computed

# whether to run experiments with different learning rules
RUN_FL = False  # similar to FLT, I prefer the local version
RUN_FIH = False  # the experiment shows it's the same...
RUN_FLT = True

# Define variables
complete_error_new_pattern = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_mean = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_new_patternFL = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_meanFL = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_new_patternFLT = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_meanFLT = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_new_patternFI = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_meanFI = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_new_patternFIH = -np.ones(shape=(TRIALS, NTRAIN))
complete_error_meanFIH = -np.ones(shape=(TRIALS, NTRAIN))

error = -np.ones(NTRAIN)
overall_error = np.zeros(NTRAIN)
overall_errorFL = np.zeros(NTRAIN)
overall_errorFLT = np.zeros(NTRAIN)
overall_errorFI = np.zeros(NTRAIN)

# summed_perturbed0 = 0
# summed_perturbed1 = 0
# summed_perturbed2 = 0

mean_w1_considered1 = 0
mean_w1_considered2 = 0


print('**Started Learning**')

for trial in range(0, TRIALS):
    if trial % eval_f == 0:
        print('Running trial ', trial+1, ' / ', TRIALS)
    solver = solverClass()
    patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, NUM_PATTERNS)
    netFisher = hopfieldNet(IMAGE_SIZE, ETA, SPARSITY)
    p = np.zeros(shape=(IMAGE_SIZE**2, IMAGE_SIZE**2))
    mean_value = SPARSITY
    original_patterns = copy.deepcopy(patterns)
    patterns = patterns - mean_value
    overall_pattern = np.zeros(shape=np.shape(patterns[:, 0]))

    for i in range(int(stored_patterns)):
        p += np.outer(patterns[:, i], patterns[:, i])
        overall_pattern += (patterns[:, i]+mean_value)
        netFisher.append_pattern(patterns[:, i], NTRAIN)
    w1 = p/70


# ========== H ========== #
    wF = w1
    for epoch in range(NTRAIN):
        diminish_lr = 2 * number_of_changed_values / (IMAGE_SIZE**2)**2
        z = diminish_lr * ETA * (np.outer(patterns[:, stored_patterns+1],
                                 patterns[:, stored_patterns+1]) - wF)
        perturbation_vector = z
        wF = wF + perturbation_vector
        netFisher.set_weights(wF)

        # checking stability of patterns after eval_epochs iterations
        overall_error = np.zeros(NTRAIN)
        for i in range(int(stored_patterns)):
            netFisher.present_pattern(original_patterns[:, i])
            netFisher.step(eval_epochs)
            output = netFisher.s
            error[epoch] = dice_coefficient(original_patterns[:, i], output)
            overall_error[epoch] += error[epoch]

        netFisher.present_pattern(original_patterns[:, stored_patterns+1])
        netFisher.step(eval_epochs)
        output = netFisher.s
        error[epoch] = dice_coefficient(original_patterns[:, stored_patterns+1], output)

        complete_error_mean[trial, epoch] = overall_error[epoch]/stored_patterns
        complete_error_new_pattern[trial, epoch] = error[epoch]

    wH_final = netFisher.w

# ========= FL ========= #
    if RUN_FL:
        # Now disturbing the weights
        wF = w1
        for epoch in range(NTRAIN):
            z = ETA * (np.outer(patterns[:, stored_patterns+1], patterns[:, stored_patterns+1]) - wF)
            netFisher.curvature = np.abs(w1)  # not an actual curvature
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
            np.fill_diagonal(weight_perturbation, 1)
            copied_curvature = copy.deepcopy(netFisher.curvature)  # not an actual curvature
            np.fill_diagonal(copied_curvature, 1000)  # setting it to a very high value such that the diagonal is not touched

            copied_curvature_tri = to_triangular(copied_curvature)  # Martino
            weight_perturbation_tri = to_triangular(weight_perturbation)  # Martino
            small_idx = np.argsort(copied_curvature_tri, axis=None)
            weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
            copied_curvature_tri[small_idx[:number_of_changed_values]] = 2000
            weight_perturbation_v2 = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)
            copied_curvature_v2 = from_triangular(IMAGE_SIZE**2, copied_curvature_tri, 1000)

            weight_perturbation = weight_perturbation_v2
            copied_curvature = copied_curvature_v2

            if (epoch % 100) == 0:
                print('The number of perturbed weights is (case 1):   ',
                      np.sum(weight_perturbation))

            xyz = np.zeros(np.shape(w1))
            xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            mean_w1_considered2 += np.mean(np.abs(xyz))
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            overall_errorFL = np.zeros(NTRAIN)
            for i in range(int(stored_patterns)):
                netFisher.present_pattern(original_patterns[:, i])
                netFisher.step(eval_epochs)
                output = netFisher.s
                error[epoch] = dice_coefficient(original_patterns[:, i], output)
                overall_errorFL[epoch] += error[epoch]

            netFisher.present_pattern(original_patterns[:, stored_patterns+1])
            netFisher.step(eval_epochs)
            output = netFisher.s
            error[epoch] = dice_coefficient(original_patterns[:, stored_patterns+1], output)

            complete_error_meanFL[trial, epoch] = overall_errorFL[epoch]/stored_patterns
            complete_error_new_patternFL[trial, epoch] = error[epoch]

            # if epoch == 0:
            #     x = np.abs(netFisher.w).flatten()
            #     netFisher.curvature = np.abs(w1)
            #     y = netFisher.curvature.flatten()
            #     plt.figure(figsize=(5,5))
            #     plt.plot(x,y, 'x')
            #     plt.xlabel('abs(w)')
            #     plt.ylabel('FisherInf')
            #     plt.show()

        wFL_final = netFisher.w

# ======== FLT ========= #
# Now disturbing the weights
    if RUN_FLT:
        wF = w1
        for epoch in range(NTRAIN):
            z = ETA * (np.outer(patterns[:, stored_patterns+1], patterns[:, stored_patterns+1]) - wF)
            netFisher.curvature = np.abs(w1)
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
            np.fill_diagonal(weight_perturbation, 1)

            weight_perturbation[np.abs(w1) < 0.008] = 1
            np.fill_diagonal(weight_perturbation, 1)
            if (epoch % 100) == 0:
                print('The number of perturbed weights is (case 2):   ',
                      np.sum(weight_perturbation))

            xyz = np.zeros(np.shape(w1))
            xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            mean_w1_considered2 += np.mean(np.abs(xyz))
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            overall_errorFLT = np.zeros(NTRAIN)
            for i in range(int(stored_patterns)):
                netFisher.present_pattern(original_patterns[:, i])
                netFisher.step(eval_epochs)
                output = netFisher.s
                error[epoch] = dice_coefficient(original_patterns[:, i], output)
                overall_errorFLT[epoch] += error[epoch]

            netFisher.present_pattern(original_patterns[:, stored_patterns+1])
            netFisher.step(eval_epochs)
            output = netFisher.s
            error[epoch] = dice_coefficient(original_patterns[:, stored_patterns+1], output)

            complete_error_meanFLT[trial, epoch] = overall_errorFLT[epoch]/stored_patterns
            complete_error_new_patternFLT[trial, epoch] = error[epoch]

            # if epoch == (NTRAIN-1):
            #    x = np.abs(netFisher.w).flatten()
            #    netFisher.curvature = np.abs(w1)
            #    y = netFisher.curvature.flatten()
            #    plt.figure(figsize=(5,5))
            #    plt.plot(x,y, 'x')
            #    plt.xlabel('abs(w)')
            #    plt.ylabel('FisherInf')
            #    plt.show()

        wFLT_final = netFisher.w


# ======== FI ======== #
# Now disturbing the weights
    wF = w1
    for epoch in range(NTRAIN):
        z = ETA * (np.outer(patterns[:, stored_patterns+1], patterns[:, stored_patterns+1]) - wF)
        netFisher.calculate_fisher_information(patterns[:, 0:stored_patterns])
        weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
        np.fill_diagonal(weight_perturbation, 1)
        copied_curvature = copy.deepcopy(netFisher.curvature)
        np.fill_diagonal(copied_curvature, 1000)  # setting it to a very high value such that the diagonal is not touched

        copied_curvature_tri = to_triangular(copied_curvature)  # Martino
        weight_perturbation_tri = to_triangular(weight_perturbation)  # Martino
        small_idx = np.argsort(copied_curvature_tri, axis=None)
        weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
        copied_curvature_tri[small_idx[:number_of_changed_values]] = 2000
        weight_perturbation_v2 = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)
        copied_curvature_v2 = from_triangular(IMAGE_SIZE**2, copied_curvature_tri, 1000)

        weight_perturbation = weight_perturbation_v2
        copied_curvature = copied_curvature_v2

        xyz = np.zeros(np.shape(w1))
        xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
        mean_w1_considered2 += np.mean(np.abs(xyz))
        perturbation_vector = weight_perturbation * z
        wF = wF + perturbation_vector
        # summed_perturbed2 += np.mean(np.abs(perturbation_vector))
        netFisher.set_weights(wF)

        overall_errorFI = np.zeros(NTRAIN)
        for i in range(int(stored_patterns)):
            netFisher.present_pattern(original_patterns[:, i])
            netFisher.step(eval_epochs)
            output = netFisher.s
            error[epoch] = dice_coefficient(original_patterns[:, i], output)
            overall_errorFI[epoch] += error[epoch]

        netFisher.present_pattern(original_patterns[:, stored_patterns+1])
        netFisher.step(eval_epochs)
        output = netFisher.s
        error[epoch] = dice_coefficient(original_patterns[:, stored_patterns+1], output)

        complete_error_meanFI[trial, epoch] = overall_errorFI[epoch]/stored_patterns
        complete_error_new_patternFI[trial, epoch] = error[epoch]

        # if epoch == (NTRAIN-1):
        #    x = np.abs(netFisher.w).flatten()
        #    netFisher.calculate_fisher_information(patterns[:,0:stored_patterns+1])
        #    y = netFisher.curvature.flatten()
        #    plt.figure(figsize=(5,5))
        #    plt.plot(x,y, 'x')
        #    plt.xlabel('abs(w)')
        #    plt.ylabel('FisherInf')
        #    plt.show()

# ======== FIH ========= #
# Now disturbing the weights using hebbian way for fisher information
    if RUN_FIH:
        wF = w1
        for epoch in range(NTRAIN):
            z = ETA * (np.outer(patterns[:, stored_patterns+1], patterns[:, stored_patterns+1]) - wF)
            netFisher.calculate_fisher_information_hebbian(patterns[:, 0:stored_patterns])
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
            np.fill_diagonal(weight_perturbation, 1)
            copied_curvature = copy.deepcopy(netFisher.curvature)
            np.fill_diagonal(copied_curvature, 1000)  # setting it to a very high value such that the diagonal is not touched

            copied_curvature_tri = to_triangular(copied_curvature)
            weight_perturbation_tri = to_triangular(weight_perturbation)
            small_idx = np.argsort(copied_curvature_tri, axis=None)
            weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
            copied_curvature_tri[small_idx[:number_of_changed_values]] = 2000
            weight_perturbation_v2 = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)
            copied_curvature_v2 = from_triangular(IMAGE_SIZE**2, copied_curvature_tri, 1000)

            weight_perturbation = weight_perturbation_v2
            copied_curvature = copied_curvature_v2

            xyz = np.zeros(np.shape(w1))
            xyz[weight_perturbation == 1] = w1[weight_perturbation == 1]
            mean_w1_considered2 += np.mean(np.abs(xyz))
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            netFisher.set_weights(wF)

            overall_errorFIH = np.zeros(NTRAIN)
            for i in range(int(stored_patterns)):
                netFisher.present_pattern(original_patterns[:, i])
                netFisher.step(eval_epochs)
                output = netFisher.s
                error[epoch] = dice_coefficient(original_patterns[:, i], output)
                overall_errorFIH[epoch] += error[epoch]

            netFisher.present_pattern(original_patterns[:, stored_patterns+1])
            netFisher.step(eval_epochs)
            output = netFisher.s
            error[epoch] = dice_coefficient(original_patterns[:, stored_patterns+1], output)

            complete_error_meanFIH[trial, epoch] = overall_errorFIH[epoch]/stored_patterns
            complete_error_new_patternFIH[trial, epoch] = error[epoch]

            # if epoch == (NTRAIN-1):
            #    x = np.abs(netFisher.w).flatten()
            #    netFisher.calculate_fisher_information(patterns[:,0:stored_patterns+1])
            #    y = netFisher.curvature.flatten()
            #    plt.figure(figsize=(5,5))
            #    plt.plot(x,y, 'x')
            #    plt.xlabel('abs(w)')
            #    plt.ylabel('FisherInf')
            #    plt.show()

        wFIH_final = netFisher.w

savedict = {
    'trad_80_O': complete_error_mean,
    'trad_80_N': complete_error_new_pattern,
    'FI_O': complete_error_meanFI,
    'FI_N': complete_error_new_patternFI}

keys = ['trad_80_O', 'trad_80_N']

if RUN_FL:
    savedict['FL_O'] = complete_error_meanFL
    savedict['FL_N'] = complete_error_new_patternFL
    keys += ['FL_O', 'FL_N']
if RUN_FLT:
    savedict['FLT_O'] = complete_error_meanFLT
    savedict['FLT_N'] = complete_error_new_patternFLT
    keys += ['FLT_O', 'FLT_N']
keys += ['FI_O', 'FI_N']
if RUN_FIH:
    savedict['FIH_O'] = complete_error_meanFIH
    savedict['FIH_N'] = complete_error_new_patternFIH
    keys += ['FIH_O', 'FIH_N']

filename = "../Complete_errors_stored{}_size{}.npz".format(stored_patterns, IMAGE_SIZE)
np.savez(filename, **savedict)

print('**Finished**')
