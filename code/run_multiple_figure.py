from hopfieldNetwork import hopfieldNet
from solverFile import solverClass
import numpy as np
import copy
from scipy.special import expit
import matplotlib.pyplot as plt

# import IPython

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


# ETA = 0.001  # learning rate
ETA = 0.01  # learning rate
ETA = 0.1  # learning rate
# ETA = 0.05  # learning rate
SPARSITY = 0.1  # number of zeros: SPARSITY = 0.1 means 10% ones and 90% zeros
IMAGE_SIZE = 10  # the size of our pattern will be (IMAGE_SIZE x IMAGE_SIZE)
eval_f = 1  # evaluation frequency (every eval_f-th iteration)
TRIALS = 10  # number of trials over which the results will be averaged
# TRIALS = 1  # number of trials over which the results will be averaged
less_changed_weight_value = 0.00
# the learning rate of weights which are considered important have a
# learning rate of ETA * less_changed_weight_value
# epochs_patterns_presented = 200
# epochs_patterns_presented = 10
epochs_patterns_presented = 10
# epochs_patterns_presented = 30
N_PRE = 10 # number of pre-trained patterns
n_stored_patterns = 0
n_new_patterns = 80 #15
# n_new_patterns = 15
n_tot_patterns = n_stored_patterns + n_new_patterns  # n of patterns created
NTRAIN = epochs_patterns_presented * n_new_patterns  # number of epochs
# number_of_changed_values = 4700
# number_of_changed_values = 4700
# number_of_changed_values = 4500
# number_of_changed_values = 4400
# number_of_changed_values_diag_0 = number_of_changed_values + IMAGE_SIZE**2/2
number_of_changed_values = IMAGE_SIZE**2*(IMAGE_SIZE**2-1)//2//100*95
print('Static weights: '+str(IMAGE_SIZE**2*(IMAGE_SIZE**2-1)/2-number_of_changed_values))
# the number of weigths that are changed is 2*number_of_changed_values
# (The factor of 2 is because of the symmetry of the weight matrix)
eval_epochs = 10  # how many steps are run before dice coefficient computed

fim_scale = 25 # # a in exp(-aF) to scale learning rates

# whether to run experiments with different learning rules
RUN_FL = True  # uses weight value, similar to FLT
RUN_FI = True  # FI
RUN_FIH = False  # Hebbian FI
RUN_FLT = True  # Thresholded FL
RUN_SCL = True  # scaled weight value
RUN_SCLT = True  # scaled weight value

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
if RUN_SCLT:
    Errors['FL_SCLT'] = -np.ones(shape=(TRIALS, NTRAIN, n_tot_patterns))


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

    NTRAIN_PRE = epochs_patterns_presented * N_PRE  # number of epochs
    pre_patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, N_PRE) - SPARSITY
    all_patterns = np.hstack((pre_patterns,patterns))

    # learning patterns
    p = np.random.random((IMAGE_SIZE**2, IMAGE_SIZE**2))
    p = np.zeros(shape=(IMAGE_SIZE**2, IMAGE_SIZE**2))
    if n_stored_patterns>0:
        for i in range(n_stored_patterns):
            p += np.outer(patterns[:, i], patterns[:, i])
            netFisher.append_pattern(patterns[:, i], NTRAIN)
        w1 = p/n_stored_patterns
        ############################## RUN PRE-TEST ##############################
        netFisher.set_weights(w1)
        overall_error = 0
        for i in range(int(n_stored_patterns)):
            netFisher.present_pattern(original_patterns[:,i])
            netFisher.step(100)
            output = netFisher.s
            error = np.sum(original_patterns[:,i]-output)**2
            overall_error += error
        print('The overall_error is:   ', overall_error) # Error should be 0
    else:
        npre = 10
        pre_patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, npre)-SPARSITY
        for i in range(npre):
            p += np.outer(pre_patterns[:, i], pre_patterns[:, i])
        w1 = p/npre
        # np.fill_diagonal(w1, 0)
    netFisher.set_weights(w1)

# ========== H ========== #
# Traditional learning rule
    wF = np.copy(w1)
    for epoch in range(NTRAIN+NTRAIN_PRE):
        id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
        pattern_taught = all_patterns[:, id_pattern_taught]

        diminish_lr = 2 * number_of_changed_values / (IMAGE_SIZE**2)**2
        z = diminish_lr * ETA * (np.outer(pattern_taught, pattern_taught) - wF)

        # training
        perturbation_vector = z
        wF = wF + perturbation_vector
        # np.fill_diagonal(wF, 0)
        netFisher.set_weights(wF)

        if (epoch % 100) == 0:
            print(epoch,'wF (25,50,75%): ',np.percentile(wF,(25,50,75)))


        # checking stability of patterns after eval_epochs iterations
        if epoch>=NTRAIN_PRE:
            Errors['trad_80_N'][trial, epoch-NTRAIN_PRE] = evaluate_stability(netFisher, original_patterns.T[:n_tot_patterns], average=False)

# ========= FL ========= #
# Now perturbing the weights using the value of the weight (quantile)
    if RUN_FL:
        wF = np.copy(w1)
        # netFisher.curvature = np.abs(wF)  # not an actual curvature
        for epoch in range(NTRAIN+NTRAIN_PRE):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = all_patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)
            netFisher.curvature = wF  # not an actual curvature
            # netFisher.curvature = np.abs(wF)  # not an actual curvature
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))

            copied_curvature_tri = to_triangular(netFisher.curvature)
            weight_perturbation_tri = to_triangular(weight_perturbation)
            small_idx = np.argsort(copied_curvature_tri, axis=None)
            weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
            #weight_perturbation_tri[small_idx[:int(number_of_changed_values_diag_0)]] = 1
            weight_perturbation = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)
            #np.fill_diagonal(weight_perturbation, 0)

            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            # np.fill_diagonal(wF, 0)
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            if epoch>=NTRAIN_PRE:
                Errors['FL_N'][trial, epoch-NTRAIN_PRE] = evaluate_stability(netFisher, original_patterns.T[:n_tot_patterns], average=False)

# ======== FLT ========= #
# Now perturbing the weights using the value of the weight (threshold)
    if RUN_FLT:
        wF = np.copy(w1)
        for epoch in range(NTRAIN+NTRAIN_PRE):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = all_patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)
            netFisher.curvature = wF
            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
            np.fill_diagonal(weight_perturbation, 1)

            # weight_perturbation[np.abs(wF) < 0.112] = 1
            # weight_perturbation[np.abs(wF) < 0.45] = 1
            weight_perturbation[wF < 0.09] = 1
            np.fill_diagonal(weight_perturbation, 1)

            # training
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            # np.fill_diagonal(wF, 0)
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            if epoch>=NTRAIN_PRE:
                Errors['FLT_N'][trial, epoch-NTRAIN_PRE] = evaluate_stability(netFisher, original_patterns.T[:n_tot_patterns], average=False)

# ======== FI ======== #
# Weights perturbed using Fisher Information
    if RUN_FI:
        wF = np.copy(w1)
        # if id_pattern_taught>0:
        #     netFisher.calculate_fisher_information(  # TODO review
        #         patterns[:, :id_pattern_taught]+SPARSITY)
        for epoch in range(NTRAIN+NTRAIN_PRE):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = all_patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)

            if id_pattern_taught>=N_PRE:
                netFisher.calculate_fisher_information(all_patterns[:, N_PRE:id_pattern_taught]+SPARSITY)
                weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(w1))
                copied_curvature_tri = to_triangular(netFisher.curvature)
                weight_perturbation_tri = to_triangular(weight_perturbation)
                small_idx = np.argsort(np.abs(copied_curvature_tri), axis=None)
                #weight_perturbation_tri[small_idx[:int(number_of_changed_values_diag_0)]] = 1
                weight_perturbation_tri[small_idx[:int(number_of_changed_values)]] = 1
                weight_perturbation = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)
            else:
                weight_perturbation = np.ones(shape=np.shape(w1))

            #np.fill_diagonal(weight_perturbation, 0)
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            # np.fill_diagonal(wF, 0)
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            if epoch>=NTRAIN_PRE:
                Errors['FI_N'][trial, epoch-NTRAIN_PRE] = evaluate_stability(netFisher, original_patterns.T[:n_tot_patterns], average=False)

# ======== FIH ========= #
# Now perturbing the weights using Hebbian way for Fisher information
    if RUN_FIH:
        wF = np.copy(w1)
        for epoch in range(NTRAIN+NTRAIN_PRE):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = all_patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)

            if id_pattern_taught>0:
                netFisher.calculate_fisher_information_hebbian(  # TODO review
                    patterns[:, :id_pattern_taught])
                version_hebbian = netFisher.curvature
                netFisher.calculate_fisher_information(  # TODO review
                    patterns[:, :id_pattern_taught])
                version_variance = netFisher.curvature

                equality_var = np.all(np.equal(version_hebbian, version_variance))

                dist = np.mean((version_hebbian-version_variance)**2)

                if dist > 1e-30 and (epoch % 100) == 0:
                    print('The mean squared error between the Hebbian and the non-Hebbian approach to calculate the Fisher ')
                    print('information has been detected to be higher than 1e-30. This should not be the case.')
                    print('Epoch:                ', epoch)
                    print('Equality var:         ', equality_var)
                    print('Euclidean Distance:   ', dist)
                    # plt.imshow(version_hebbian)
                    # plt.show()
                    # plt.imshow(version_variance)
                    # plt.show()

                netFisher.calculate_fisher_information_hebbian(  # TODO review
                    patterns[:, :id_pattern_taught])

            weight_perturbation = less_changed_weight_value*np.ones(shape=np.shape(wF))

            copied_curvature_tri = to_triangular(netFisher.curvature)
            weight_perturbation_tri = to_triangular(weight_perturbation)
            small_idx = np.argsort(copied_curvature_tri, axis=None)
            weight_perturbation_tri[small_idx[:number_of_changed_values]] = 1
            weight_perturbation = from_triangular(IMAGE_SIZE**2, weight_perturbation_tri, 1)

            # training
            perturbation_vector = weight_perturbation * z
            wF = wF + perturbation_vector
            # np.fill_diagonal(wF, 0)
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            if epoch>=NTRAIN_PRE:
                Errors['FIH_N'][trial, epoch-NTRAIN_PRE] = evaluate_stability(netFisher, original_patterns.T[:n_tot_patterns], average=False)

# ========= SCL ========= #
# Now perturbing the weights proportional to its value
    if RUN_SCL:
        wF = np.copy(w1)
        for epoch in range(NTRAIN+NTRAIN_PRE):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = all_patterns[:, id_pattern_taught]

            z = ETA * (np.outer(pattern_taught, pattern_taught) - wF)
            netFisher.curvature = np.abs(wF)  # not an actual curvature
            #netFisher.curvature = np.abs(wF*(1-wF)-4*SPARSITY*wF)  # not an actual curvature
            # netFisher.curvature = np.abs(wF-4*SPARSITY*wF)  # not an actual curvature
            # netFisher.curvature = np.abs(wF-np.mean(wF))  # not an actual curvature
            # netFisher.curvature = wF  # not an actual curvature
            weight_perturbation = np.exp(-netFisher.curvature*fim_scale)

            if (epoch % 100) == 0:
                print(epoch,'perturbation (25,50,75% , min, max, mean(wF)): ',np.percentile(weight_perturbation,(25,50,75)),np.min(weight_perturbation), np.max(weight_perturbation),np.mean(wF)  )
            # IPython.embed()
            # training
            # if (epoch % 100) == 0:
            #     # print('The w is:   ', np.sum(wF))
            #     print('The sum is:   ', np.sum(weight_perturbation))
            perturbation_vector = weight_perturbation * z #*5
            wF = wF + perturbation_vector
            # np.fill_diagonal(wF, 0)
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            if epoch>=NTRAIN_PRE:
                Errors['FL_SCL'][trial, epoch-NTRAIN_PRE] = evaluate_stability(netFisher, original_patterns.T[:n_tot_patterns], average=False)

# ========= SCLT ========= #
# Now perturbing the weights proportional to its value + threshold
    if RUN_SCLT:
        wF = np.copy(w1)
        for epoch in range(NTRAIN+NTRAIN_PRE):
            id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
            pattern_taught = all_patterns[:, id_pattern_taught]

            z = (np.outer(pattern_taught, pattern_taught) - wF)
            netFisher.curvature = np.abs(wF)  # not an actual curvature
            # netFisher.curvature = wF  # not an actual curvature
            # netFisher.curvature = np.abs(wF-4*SPARSITY*wF)  # not an actual curvature
            weight_perturbation = np.exp(-netFisher.curvature*fim_scale)

            # if (epoch % 100) == 0:
            #     print('perturbation (25,50,75% , min, max): ',np.percentile(weight_perturbation,(25,50,75)),np.min(weight_perturbation), np.max(weight_perturbation)  )

            # training
            if (epoch % 100) == 0:
                print('The sum is:   ', np.sum(weight_perturbation))
            perturbation_vector = weight_perturbation * z * (np.abs(z)>0.19)
            wF = wF + ETA * perturbation_vector
            # np.fill_diagonal(wF, 0)
            netFisher.set_weights(wF)

            # checking stability of patterns after eval_epochs iterations
            if epoch>=NTRAIN_PRE:
                Errors['FL_SCLT'][trial, epoch-NTRAIN_PRE] = evaluate_stability(netFisher, original_patterns.T[:n_tot_patterns], average=False)


filename = "../Complete_errors_stored{}_size{}_spars{}_epochs{}_eta{}_ALL.npz".format(
    n_tot_patterns, IMAGE_SIZE, SPARSITY, epochs_patterns_presented, ETA)
np.savez(filename, **Errors)
print(filename)
print('**Finished**')