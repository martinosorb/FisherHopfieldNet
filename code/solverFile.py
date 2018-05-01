import numpy as np
from hopfieldNetwork import hopfieldNet
import matplotlib.pyplot as plt
import copy
import random

class solverClass(object):
    """docstring for solver."""
    def __init__(self):
        pass

    def create_patterns(self, sparsity, image_size, numPatterns):
        #sparse = 0.3
        patterns = np.zeros(shape = (image_size**2, numPatterns))
        x = int(round(image_size**2*sparsity))
        for i in range(numPatterns):
            r = random.sample(range(0, image_size**2-1), x)
            r = np.asarray(r)
            patterns[r,i] = 1
        return patterns

    def learn_pattern(self, net, pattern, NTRAIN, fisher):
        net.pattern_matrix = np.c_[net.pattern_matrix,pattern]
        [neurons, numPatterns] = np.shape(net.pattern_matrix)
        net.training_length += NTRAIN
        new_energy_pattern = np.zeros(shape = (net.training_length, numPatterns))
        new_energy_pattern[0:net.training_length - NTRAIN, 0:numPatterns-1] = net.energy_pattern
        net.energy_pattern = new_energy_pattern

        every_nth = 1
        my_energy_pattern1 = np.zeros(20*NTRAIN)
        net.memorized_weights = net.w
        new_pattern_learnt = False

        summed_num_forgotten = 0
        summed_av_error = 0

        num_forgotten = -1
        av_error = -1

        if fisher == True:
            print('Results when adapting weights using Fisher Information:')
            if numPatterns == 2:
                z = np.random.rand(*np.shape(net.curvature))
                untouched_weight_perc = 0.4
                z[z<untouched_weight_perc] = 0
                z[z>untouched_weight_perc] = 1
                net.curvature = z
            if numPatterns > 2:

                net.calculate_fisher_information(net.pattern_matrix[:,0:numPatterns-1])
                #print('NOTE: FISHER INFORMATION IS COMMENTED OUT')
                print('Number of small values in curvature: ', len(np.where(net.curvature < 0.9)[0]))

                # x = np.abs(net.w).flatten()
                # y = net.curvature.flatten()
                #y = x**2
                # plt.figure(figsize=(5,5))
                # plt.plot(x,y, 'o')
                # plt.xlabel('abs(w)')
                # plt.ylabel('FisherInf')
                # plt.show()

                #print(np.amax(net.curvature))
                #print(np.amin(net.curvature))
            for i in range(net.training_length-NTRAIN, net.training_length):
                [neurons, numPatterns] = np.shape(net.pattern_matrix)
                net.present_pattern(net.pattern_matrix[:,numPatterns-1])#want to store the last pattern
                if np.isnan(net.w).all() == False and np.isnan(net.curvature).all() == False:
                    #self.w += self.eta * self.curvature * (np.outer(self.s, self.s) - self.w)
                    x = np.outer(net.s, net.s)
                    #self.w += self.eta * self.curvature * (np.outer(self.s, self.s) - self.w)
                    z = np.abs(net.memorized_weights)
                    z = (0.25 - z)**2
                    z = z / np.mean(z)


                    #produces super-many zero values and hence slows down learning
                    z = np.abs(net.memorized_weights)
                    z = np.amax(z) - z
                    #z = z / np.mean(z)
                    #print('non-inverse: ', len(np.where(z == 0)[0]))

                    #net.w += net.eta * z * (x-net.w)
                    #print('Curvature as learning rule')
                    net.w += net.eta * net.curvature * (x-net.w)
                    #net.w += net.eta * (np.abs(x) * 20)**(1/2)  * net.curvature * (x-net.w)

                    d = np.sum((net.w - x)**2)
                    print('Distance measure between optimal and current: ', d)

                    if i == net.training_length - 1:
                        print('It is time to change the situation dadadadam')
                        print('Old weight matrix:           ', net.w)
                        net.w = x
                        print('New/correct weight matrix:   ', net.w)

                    if i % every_nth == 0:
                        new_pattern_learnt, num_forgotten, av_error = self.evaluation_intermediate(net)
                        #print('av_error now  = ', av_error)
                        if new_pattern_learnt == True:
                            print('Learning is done. It took ', i - (net.training_length-NTRAIN), ' iterations.')
                            break
                    # if i % 1000 == 0:
                        #print('inverse: ', len(np.where(z == 0)[0]))
                        #print('the average value of the abs of all weights:   ', np.mean(np.abs(net.w)))
                else:
                    print('Problem in perturb_important_weights')
                    if np.isnan(net.curvature).any() == True:
                        print('curvature is the problem')

                for j in range(numPatterns):
                    net.present_pattern(net.pattern_matrix[:,j])
                    net.energy_pattern[i,j] = net.calculate_energy()
        else:
            print('Results when using the Hebbian Learning Rule:')
            for i in range(net.training_length-NTRAIN, net.training_length):
                [neurons, numPatterns] = np.shape(net.pattern_matrix)
                net.s = net.pattern_matrix[:,numPatterns-1]#want to store the 4th pattern
                if np.isnan(net.w).all() == False and np.isnan(net.curvature).all() == False:
                    net.w += net.eta * (np.outer(net.s, net.s) - net.w)
                    x = np.outer(net.s, net.s)
                    d = np.sum((net.w - x)**2)
                    if i == net.training_length - 1:
                        print('It is time to change the situation dadadadam')
                        net.w = x
                    if i % every_nth == 0:
                        new_pattern_learnt, num_forgotten, av_error = self.evaluation_intermediate(net)
                        if new_pattern_learnt == True:
                            print('Learning is done. It took ', i - (net.training_length-NTRAIN), ' iterations.')
                            break
                else:
                    print('Problem in perturb_weights')
                    if np.isnan(net.curvature).any() == True:
                        print('curvature is the problem')
                # if i % 1000 == 0:
                #     print('the average value of the abs of all weights:   ', np.mean(np.abs(net.w)))
                for j in range(numPatterns):
                    net.present_pattern(net.pattern_matrix[:,j])
                    net.energy_pattern[i,j] = net.calculate_energy()
        #self.evaluation_accuracy(net)
        #print('\n\n')
        return net.energy_pattern, num_forgotten, av_error

    def plot_energy(self, energy_pattern_fisher, energy_pattern, stored_p, overall_p, NTRAIN_prev):
        [NTRAIN, numPatterns] = np.shape(energy_pattern)
        plt.figure(figsize=(30,10))
        max_value1 = np.amax(-energy_pattern_fisher)
        max_value2 = np.amax(-energy_pattern)
        max_value = np.amax([max_value1, max_value2])

        # E_f = np.sum(energy_pattern_fisher[-1,:])
        # E_h = np.sum(energy_pattern[-1,:])
        # print('Energy Fisher:  ', E_f)
        # print('Energy Hebbian: ', E_h)

        for i in range(numPatterns):
            plt.subplot(1,numPatterns,i+1)
            energy12, = plt.plot(-energy_pattern_fisher[:,i], label='Fisher information')
            energy22, = plt.plot(-energy_pattern[:,i], label='Hebbian Learning ')
            plt.xlabel('Iteration')
            plt.ylabel('-Energy')
            plt.title('Pattern %i'%(i+1))
            plt.xlim([stored_p*NTRAIN_prev, overall_p*NTRAIN_prev])
            plt.ylim([0, max_value])
            plt.legend(handles=[energy12, energy22])
        plt.show()

    def evaluation_noise(self, net, flip_ratio):
        x = np.random.rand(*np.shape(net.pattern_matrix))
        original_patterns = net.pattern_matrix + net.sparsity
        perturbed_patterns = copy.deepcopy(net.pattern_matrix)
        perturbed_patterns[x < flip_ratio] = -perturbed_patterns[x < flip_ratio]
        [neurons, numPatterns] = np.shape(perturbed_patterns)
        dist = 1e10 * np.ones(numPatterns)
        perc = np.zeros(numPatterns)
        for pattern in range(numPatterns):
            net.present_pattern(perturbed_patterns[:,pattern])
            net.step(100)
            output_pattern = net.s
            dist[pattern] = np.mean((output_pattern - original_patterns[:,pattern])**2)
            perc[pattern] = np.sum((output_pattern - original_patterns[:,pattern]) != 0)
            #print('Pattern ', pattern+1, ', flip-ratio ', flip_ratio, ',
            #Percentage of error ', perc[pattern], '%')
        return perc

    def evaluation_plot(self, net):
        [neurons, numPatterns] = np.shape(net.pattern_matrix)
        flip_ratios = np.linspace(0,0.5,21)
        index = 0
        all_dists = np.zeros(shape = (numPatterns, len(flip_ratios)))
        plt.figure(figsize=(30,10))
        for flip_ratio in flip_ratios:
            percentage = self.evaluation_noise(net.pattern_matrix, flip_ratio)
            all_dists[:,index] = percentage
            index += 1
        for regarded_pattern in range(numPatterns):
            plt.subplot(1,numPatterns,regarded_pattern+1)
            plt.xlabel('Flips [%]')
            plt.ylabel('Error [%]')
            plt.title('Pattern %i' %(regarded_pattern+1))
            plt.plot(flip_ratios*100, all_dists[regarded_pattern,:])
        plt.show()

    def evaluation_accuracy(self, net):
        [neurons, numPatterns] = np.shape(net.pattern_matrix)
        flip_ratios = np.linspace(0,0.5,21)
        index = 0
        all_dists = np.zeros(shape = (numPatterns, len(flip_ratios)))
        for flip_ratio in flip_ratios:
            percentage = self.evaluation_noise(net, flip_ratio)
            all_dists[:,index] = percentage
            index += 1
        current_score = all_dists[:,0]#score on undisturbed picture as input
        if numPatterns > 1:
            print('Average error of previously stored ', numPatterns-1, ' patterns:   ', np.mean(current_score[0:numPatterns-1])/neurons*100, '%')
        print('Error on recently learnt pattern:   ', current_score[numPatterns-1]/neurons*100, '%')
        # if np.mean(current_score) == 0:
        #     print('WIIIIIN')

    def evaluation_intermediate(self, net):
        [neurons, numPatterns] = np.shape(net.pattern_matrix)
        flip_ratios = np.linspace(0,0.5,21)
        flip_ratios = [0, 0.1]
        index = 0
        pattern_learnt = False
        all_dists = np.zeros(shape = (numPatterns, len(flip_ratios)))
        num_forgotten = -1
        av_error = -1
        #print('Setting av_error to -1 in evaluation_intermediate')
        for flip_ratio in flip_ratios:
            percentage = self.evaluation_noise(net, flip_ratio)
            all_dists[:,index] = percentage
            index += 1
        current_score = all_dists[:,0]#score on undisturbed picture as input
        if current_score[numPatterns-1]/neurons*100 == 0:
            if numPatterns > 1:
                av_error = np.mean(current_score[0:numPatterns-1])/neurons*100
                print('Average error of previously stored ', numPatterns-1, ' patterns:   ', av_error, '%')
                num_forgotten = np.sum(current_score[0:numPatterns-1] != 0)
                print('Number of forgotten patterns:                       ', num_forgotten)
                if num_forgotten != 0:
                    zzz = current_score[0:numPatterns-1] != 0
                    zzz = np.asarray(zzz)
                    a = np.linspace(1,numPatterns-1,numPatterns-1)
                    b = a[zzz != 0]
                    #print('indizes: ', np.where((current_score[0:numPatterns-1] != 0) == True))
                    print('Indizes of forgotten elements:                      ', b)
            print('Error on recently learnt pattern:                   ', current_score[numPatterns-1]/neurons*100, '%')
            pattern_learnt = True
        return pattern_learnt, num_forgotten, av_error
        # if np.mean(current_score) == 0:
        #     print('WIIIIIN')























    def train_pattern1(self, energy_pattern, dists, NTRAIN, patterns):
        for i in range(NTRAIN//2):
            self.net.present_pattern(patterns[:,0])
            self.net.learn()
            energy_pattern[i] = self.net.calculate_energy()
        return energy_pattern, dists

    def train_pattern123(self, energy_pattern, dists, NTRAIN, patterns):
        for i in range(NTRAIN//2):
            if i % 3 == 0:
                self.net.present_pattern(patterns[:,0])
            elif i % 3 == 1:
                self.net.present_pattern(patterns[:,1])
            elif i % 3 == 2:
                self.net.present_pattern(patterns[:,2])
            self.net.learn()

            self.net.present_pattern(patterns[:,0])
            energy_pattern[i,0] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,1])
            energy_pattern[i,1] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,2])
            energy_pattern[i,2] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,3])
            energy_pattern[i,3] = self.net.calculate_energy()
        return energy_pattern, dists

    def perturb_pattern(self, energy_pattern, dists, NTRAIN, pattern1, importance):
        for i in range(NTRAIN//2, NTRAIN):
            self.net.calculate_curvature()
            if importance == True:
                self.net.perturb_important_weights()
            else:
                self.net.perturb_unimportant_weights()
            energy_pattern[i] = self.net.calculate_energy()
            #dists1[i] = np.sum((net.w - target_w1)**2)

        return energy_pattern, dists

    def perturb_pattern_normally(self, energy_pattern, dists, NTRAIN, patterns):
        for i in range(NTRAIN//2, NTRAIN):
            self.net.perturb_weights(patterns)
            energy_pattern[i] = self.net.calculate_energy()

            self.net.present_pattern(patterns[:,0])
            energy_pattern[i,0] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,1])
            energy_pattern[i,1] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,2])
            energy_pattern[i,2] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,3])
            energy_pattern[i,3] = self.net.calculate_energy()

        return energy_pattern, dists

    def perturb_pattern_Fisher(self, energy_pattern, dists, NTRAIN, patterns, importance, ETA_unlearn, eval_f):
        self.net.calculate_fisher_information(patterns[:,0:3])
        print('This works')
        for i in range(NTRAIN//2, NTRAIN):
            #self.net.calculate_fisher_information(patterns)
            if importance == True:
                self.net.perturb_important_weights(patterns)
            else:
                self.net.perturb_unimportant_weights(patterns, ETA_unlearn)

            self.net.present_pattern(patterns[:,0])
            energy_pattern[i,0] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,1])
            energy_pattern[i,1] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,2])
            energy_pattern[i,2] = self.net.calculate_energy()
            self.net.present_pattern(patterns[:,3])
            energy_pattern[i,3] = self.net.calculate_energy()
            #dists1[i] = np.sum((net.w - target_w1)**2)

            # if i % eval_f == 0:
            #     print('Evaluating after ', i, ' steps. Disturbing important = ', importance)
            #     self.evaluation_accuracy(patterns)
            #     overall_energy = 0
            #     for p in range(4):
            #         overall_energy += energy_pattern[i,p]
            #     print('Overall energy: E = ', overall_energy)

        return energy_pattern, dists

    def plot_energy_pattern(self, energy_pattern1, energy_pattern2, energy_pattern3):
        #usually energy_pattern1 will have been obtained by using the fisher information
        plt.figure(figsize=(30,10))
        plt.subplot(1,4,1)
        energy11, = plt.plot(energy_pattern1[:,0], label='E_p1 important w disturbed')
        energy12, = plt.plot(energy_pattern2[:,0], label='E_p1 unimportant w disturbed')
        energy13, = plt.plot(energy_pattern3[:,0], label='E_p1 normal w disturbed')
        plt.xlabel('Iteration')
        plt.ylabel('-Energy')
        plt.title('Pattern 1')
        #plt.xlim([19500, 20500])
        plt.ylim([0, 10])
        plt.legend(handles=[energy11, energy12, energy13])
        plt.subplot(1,4,2)
        energy21, = plt.plot(energy_pattern1[:,1], label='E_p2 important w disturbed')
        energy22, = plt.plot(energy_pattern2[:,1], label='E_p2 unimportant w disturbed')
        energy23, = plt.plot(energy_pattern3[:,1], label='E_p2 normal w disturbed')
        plt.xlabel('Iteration')
        plt.ylabel('-Energy')
        plt.title('Pattern 2')
        #plt.xlim([19500, 20500])
        plt.ylim([0, 10])
        plt.legend(handles=[energy21, energy22, energy23])
        plt.subplot(1,4,3)
        energy31, = plt.plot(energy_pattern1[:,2], label='E_p3 important w disturbed')
        energy32, = plt.plot(energy_pattern2[:,2], label='E_p3 unimportant w disturbed')
        energy33, = plt.plot(energy_pattern3[:,2], label='E_p3 normal w disturbed')
        plt.xlabel('Iteration')
        plt.ylabel('-Energy')
        plt.title('Pattern 3')
        #plt.xlim([19500, 20500])
        plt.ylim([0, 10])
        plt.legend(handles=[energy31, energy32, energy33])
        plt.subplot(1,4,4)
        energy41, = plt.plot(energy_pattern1[:,3], label='E_p4 important w disturbed')
        energy42, = plt.plot(energy_pattern2[:,3], label='E_p4 unimportant w disturbed')
        energy43, = plt.plot(energy_pattern3[:,3], label='E_p4 normal w disturbed')
        plt.xlabel('Iteration')
        plt.ylabel('-Energy')
        plt.title('Pattern 4')
        #plt.xlim([19500, 20500])
        plt.ylim([0, 10])
        plt.legend(handles=[energy41, energy42, energy43])
        plt.show()
