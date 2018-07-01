import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class hopfieldNet(object):
    def __init__(self, n, eta, sparsity):
        self.n = n
        self.N = n**2
        self.w = np.random.normal(size=[self.N, self.N])*0.1
        self.memorized_weights = np.random.normal(size=[self.N, self.N])
        self.s = np.zeros(self.N, dtype=bool)
        self.eta = eta
        self.curvature = np.ones(shape = (self.N, self.N))
        self.expectation = np.zeros(shape = (self.N, self.N))
        self.variance = np.zeros(shape = (self.N, self.N))
        self.energy_factor = 1000
        self.sparsity = sparsity
        self.image_size = len(self.s)
        self.pattern_matrix = np.zeros(shape = (self.image_size,0))
        self.training_length = 0
        self.energy_pattern = np.zeros(shape = (0,0))


    def append_pattern(self, pattern, NTRAIN):
        self.pattern_matrix = np.c_[self.pattern_matrix,pattern]
        [neurons, numPatterns] = np.shape(self.pattern_matrix)
        self.training_length += NTRAIN
        new_energy_pattern = np.zeros(shape = (self.training_length, numPatterns))
        new_energy_pattern[0:self.training_length - NTRAIN, 0:numPatterns-1] = self.energy_pattern
        self.energy_pattern = new_energy_pattern

    def step(self, k=1, bias=1):
        #print('sparsity:  ', self.sparsity)
        for _ in range(k):
            probs = 0.5 + 0.5 * np.sign((self.w @ self.s) - bias)
            # probs = 0.5 + 0.5 * np.sign((self.w @ self.s) - self.sparsity)
            #probs = 0.5 + 0.5 * np.sign(self.w @ self.s)
            self.present_pattern(probs)

    def present_pattern(self, xi):
        self.s = xi

    def present_random(self, p=.5):
        self.s = np.random.random(size=self.N) < p

    def learn(self):
        if np.isnan(self.w).all() == False:
            self.w += self.eta * (np.outer(self.s, self.s) - self.w)
        else:
            print('Problem in learn')

    def set_weights(self, matrix):
        self.w = matrix

    def calculate_energy(self):
        energy = -np.dot(np.dot(np.transpose(self.s), self.w/np.mean(np.abs(self.w))), self.s)
        energy = energy / self.energy_factor
        return energy

    def calculate_pattern_energies(self, patterns):
        # Introduced by martino for faster computation
        energy = -np.dot(np.dot(np.transpose(patterns),
                         self.w/np.mean(np.abs(self.w))), patterns)
        return np.diagonal(energy) / self.energy_factor

    def calculate_curvature(self):
        curvature_matrix = np.multiply(np.outer(self.s, self.s), np.outer(self.s, self.s))
        curvature_value = -np.dot(np.dot(np.transpose(self.s), self.w), self.s)
        curvature_value = np.exp(-curvature_value-np.amax(-curvature_value))
        self.curvature = np.absolute(curvature_matrix * curvature_value)
        self.curvature = self.curvature / np.mean(self.curvature)

    # def calculate_fisher_information(self, patterns):
    #     #patterns = patterns-self.sparsity
    #     [length_of_pattern, numPatterns] = np.shape(patterns)
    #     #print('Number of patterns fisher info: ',numPatterns)
    #     self.calculate_expectation(patterns)
    #     Z = self.calculate_Z(patterns)
    #     self.variance = np.zeros(shape = (self.N, self.N))
    #     smallest_energy = 1e10
    #     learnt = 0
    #     for pattern in range(numPatterns):
    #         self.present_pattern(patterns[:,pattern])
    #         energy = self.calculate_energy()
    #         if energy < smallest_energy:
    #             smallest_energy = energy
    #         exp_energy = np.exp(-energy)
    #         p = exp_energy/Z
    #         if p > 1/numPatterns * 0.75:
    #             learnt += 1
    #     for pattern in range(numPatterns):
    #         self.present_pattern(patterns[:,pattern])
    #         energy = self.calculate_energy()
    #         if energy < smallest_energy:
    #             smallest_energy = energy
    #         exp_energy = np.exp(-energy)
    #         p = exp_energy/Z
    #         if p > (1/learnt)*1.05:
    #             p = (1/learnt)*1.05
    #         #print('pattern:  ', pattern, 'has a p = ', p)
    #         value_matrix = np.outer(self.s, self.s)
    #         self.variance += p * (value_matrix - self.expectation)**2
    #     self.curvature = self.variance
    #     self.curvature = self.curvature / np.mean(self.curvature)
    #     self.present_pattern(patterns[:,0])

    def calculate_fisher_information(self, patterns):
        # energies = self.calculate_pattern_energies(patterns)
        # expenergies = np.exp(-energies)
        # Z = expenergies.sum()

        # Assuming EQUAL probability
        val_matrix = np.einsum('ij,kj->jik', patterns, patterns)
        var = np.var(val_matrix, axis=0)
        self.curvature = var/np.mean(var)

    def calculate_complete_fisher_information(self, patterns):
        [length_of_pattern, numPatterns] = np.shape(patterns)
        val_matrix = np.einsum('ij,kj->jik', patterns, patterns)
        val_some = np.asarray([val[np.triu_indices(
            length_of_pattern, 1)] for val in val_matrix])
        return(np.cov(val_some, rowvar=0))


    # def calculate_fisher_information(self, patterns):
    #     #patterns = patterns-self.sparsity
    #     [length_of_pattern, numPatterns] = np.shape(patterns)
    #     #print('Number of patterns fisher info: ',numPatterns)
    #     self.calculate_expectation(patterns)
    #     Z = self.calculate_Z(patterns)
    #     self.variance = np.zeros(shape = (self.N, self.N))
    #     smallest_energy = 1e10
    #     for pattern in range(numPatterns):
    #         self.present_pattern(patterns[:,pattern])
    #         energy = self.calculate_energy()
    #         if energy < smallest_energy:
    #             smallest_energy = energy
    #         exp_energy = np.exp(-energy)
    #         p = 1/numPatterns
    #         #print('pattern:  ', pattern, 'has a p = ', p)
    #         value_matrix = np.outer(self.s, self.s)
    #         self.variance += p * (value_matrix - self.expectation)**2
    #     self.curvature = self.variance
    #     self.curvature = self.curvature / np.mean(self.curvature)
    #     self.present_pattern(patterns[:,0])

    def calculate_fisher_information_hebbian(self, patterns):
        #patterns = patterns-self.sparsity
        sparsity = 0.1
        patterns = patterns + sparsity
        [length_of_pattern, numPatterns] = np.shape(patterns)
        #print('Number of patterns fisher info: ',numPatterns)
        self.variance = np.zeros(shape = (self.N, self.N))
        w1 = np.zeros(shape = (self.N, self.N))
        w2 = np.zeros(shape = (self.N, self.N))
        perturbed_patterns = patterns * (2*sparsity - 1)
        for pattern in range(numPatterns):
            self.present_pattern(perturbed_patterns[:,pattern]-sparsity**2)
            w1 += np.outer(self.s, self.s)
        w1 /= numPatterns
        for pattern in range(numPatterns):
            self.present_pattern(patterns[:,pattern]-sparsity)
            w2 += np.outer(self.s, self.s)
        w2 /= numPatterns
        w2 = w2 * w2
        self.curvature = w1 - w2
        self.curvature = self.curvature / np.mean(self.curvature)
        self.present_pattern(patterns[:,0])

    def calculate_expectation(self,patterns):
        [length_of_pattern, numPatterns] = np.shape(patterns)
        Z = self.calculate_Z(patterns)
        self.expectation = np.zeros(shape = (self.N, self.N))
        for pattern in range(numPatterns):
            self.present_pattern(patterns[:,pattern])
            energy = self.calculate_energy()
            exp_energy = np.exp(-energy)
            p = exp_energy/Z
            p = 1/numPatterns
            self.expectation += (p * np.outer(self.s, self.s))
        #self.expectation = 0.0

    def calculate_Z(self, patterns):
        [length_of_pattern, numPatterns] = np.shape(patterns)
        Z = 0
        for pattern in range(numPatterns):
            self.present_pattern(patterns[:,pattern])
            energy = self.calculate_energy()
            Z += np.exp(-energy)
        return Z
