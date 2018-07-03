from hopfieldNetwork import hopfieldNet
from solverFile import solverClass
import numpy as np
import copy
from scipy.special import expit

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


ETA = 0.1  # learning rate
SPARSITY = 0.1  # number of zeros: SPARSITY = 0.1 means 10% ones and 90% zeros
IMAGE_SIZE = 10  # the size of our pattern will be (IMAGE_SIZE x IMAGE_SIZE)
print(IMAGE_SIZE**2*0.16)
eval_f = 1  # evaluation frequency (every eval_f-th iteration)
TRIALS = 1  # number of trials over which the results will be averaged
less_changed_weight_value = 0.00
# the learning rate of weights which are considered important have a
# learning rate of ETA * less_changed_weight_value
epochs_patterns_presented = 5#2
n_stored_patterns = 0
n_new_patterns = 60
n_tot_patterns = n_stored_patterns + n_new_patterns  # n of patterns created
NTRAIN = epochs_patterns_presented * n_new_patterns  # number of epochs
number_of_changed_values = 4600
number_of_changed_values = 4500
number_of_changed_values = int(IMAGE_SIZE*(IMAGE_SIZE-1)/2)-1200
number_of_changed_values = IMAGE_SIZE**2*(IMAGE_SIZE**2-1)//2//100*95
print(IMAGE_SIZE**2*(IMAGE_SIZE**2-1)//2,number_of_changed_values)
# number_of_changed_values_diag_0 = number_of_changed_values + IMAGE_SIZE**2/2
# the number of weigths that are changed is 2*number_of_changed_values
# (The factor of 2 is because of the symmetry of the weight matrix)
eval_epochs = 10  # how many steps are run before dice coefficient computed

# preparing patterns
solver = solverClass()
patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, n_tot_patterns)
netFisher = hopfieldNet(IMAGE_SIZE, ETA, SPARSITY)
original_patterns = copy.deepcopy(patterns)
patterns = patterns - SPARSITY

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
    ns = 20
    pre_patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, ns) - SPARSITY
    for i in range(ns):
        p += np.outer(pre_patterns[:, i], pre_patterns[:, i])
    w1 = p/ns/1
netFisher.set_weights(w1)

##########################################################
bayes = lambda wF, c, z, t: c/(c+np.abs(wF))
bayes_thres = lambda wF, c, z, t: c/(c+np.abs(wF))*(z>t)
w_thres  = lambda wF, c, z, t: wF<c
hopfield  = lambda wF, c, z, t: np.ones(wF.shape)

def compute_dice(ETA, c, t, func):#, t = 0.0):
    wF = np.copy(w1)
    DICE = -np.ones(shape=(NTRAIN, n_tot_patterns))
    N_PRE = 0
    NTRAIN_PRE = epochs_patterns_presented * N_PRE  # number of epochs
    pre_patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, N_PRE)
#     patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, n_tot_patterns)
#     original_patterns = copy.deepcopy(patterns)
#     patterns = patterns-SPARSITY
    all_patterns = np.hstack((pre_patterns - SPARSITY,patterns))
    w_mean = np.zeros(NTRAIN+NTRAIN_PRE)
    for epoch in range(NTRAIN+NTRAIN_PRE):
        id_pattern_taught = n_stored_patterns+epoch//epochs_patterns_presented
        pattern_taught = all_patterns[:, id_pattern_taught]

        z = (np.outer(pattern_taught, pattern_taught) - wF)

        perturbation_vector = func(wF, c, np.abs(z), t) * z * ETA
        wF = wF + perturbation_vector
        netFisher.set_weights(wF)
        w_mean[epoch] = np.mean(wF)

        # checking stability of patterns after eval_epochs iterations
        if epoch>=NTRAIN_PRE:
            DICE[epoch-NTRAIN_PRE] = evaluate_stability(  # old patterns
                netFisher, original_patterns.T[:n_tot_patterns], average=False)
    return DICE

def compute_dice_error(par,args):
    if len(par)==3:
        ETA, c, t = par
    elif len(par)==2:
        ETA, c, t = par[0], par[1], 0
    else:
        ETA, c, t = par[0], 0, 0
    return -np.sum(compute_dice(ETA, c, t, args)>0.9)

###################################################

from scipy import optimize

experiments = {
    'Hopfield': (hopfield, (0.1)),
    'Weight thresh.': (w_thres, (0.1, 0.1)),
    'Bayes update': (bayes, (0.6, 0.007)),
    'Bayes with thres.': (bayes_thres, (0.1, 0.1, 0.1))
}

data = {}

for e in experiments:
    print(e)

    n_new_patterns = 40
    n_tot_patterns = n_stored_patterns + n_new_patterns
    NTRAIN = epochs_patterns_presented * n_new_patterns
    patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, n_tot_patterns)
    original_patterns = copy.deepcopy(patterns)
    patterns = patterns - SPARSITY

    exp = experiments[e]
    par = optimize.fmin(compute_dice_error, exp[1], args=(exp[0],), disp=True)

    print(par)
    storer = []
    storer.append(par)

    n_new_patterns = 60
    n_tot_patterns = n_stored_patterns + n_new_patterns
    NTRAIN = epochs_patterns_presented * n_new_patterns

    if len(par)==3:
        ETA, c, t = par
    elif len(par)==2:
        ETA, c, t = par[0], par[1], 0
    else:
        ETA, c, t = par[0], 0, 0

    dice = []
    for i in range(10):

        patterns = solver.create_patterns(SPARSITY, IMAGE_SIZE, n_tot_patterns)
        original_patterns = copy.deepcopy(patterns)
        patterns = patterns - SPARSITY

        # data = compute_dice(c[0], c[1], c[2])
        # data = compute_dice(c[0], c[1], c[2], bayes_thres)
        # data = compute_dice(c[0], c[1], 1, w_thres)
        dice.append(compute_dice(ETA, c, t, exp[0]))

    storer.append(dice)
    data[e] = storer

np.savez('../optimised_models_40.npz', **data)
