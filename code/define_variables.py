complete_error_new_pattern = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_mean = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_new_patternFL = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_meanFL = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_new_patternFLT = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_meanFLT = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_new_patternFI = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_meanFI = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_new_patternFIH = -np.ones(shape = (TRIALS, NTRAIN))
complete_error_meanFIH = -np.ones(shape = (TRIALS, NTRAIN))

error = -np.ones(NTRAIN)
overall_error = np.zeros(NTRAIN)
overall_errorFL = np.zeros(NTRAIN)
overall_errorFLT = np.zeros(NTRAIN)
overall_errorFI = np.zeros(NTRAIN)

summed_perturbed0 = 0
summed_perturbed1 = 0
summed_perturbed2 = 0

mean_w1_considered1 = 0
mean_w1_considered2 = 0
