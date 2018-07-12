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
