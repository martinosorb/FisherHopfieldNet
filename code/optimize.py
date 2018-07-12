parameters = []
num_evaluations = 0
order_dict = {}
if USE_HOPFIELD == True:
    parameters.append(optimize.fmin(compute_dice_error,(0.1,), args=(hopfield,), disp=True))
    order_dict["hopfield"] = num_evaluations
    num_evaluations += 1
if USE_EXPONENTIAL == True:
    parameters.append(optimize.fmin(compute_dice_error,(0.1, 30),args=(expo,),disp=True))
    order_dict["exponential"] = num_evaluations
    num_evaluations += 1
if USE_EXPONENTIAL_THRESHOLD == True:
    parameters.append(optimize.fmin(compute_dice_error,(0.1, 10, 0.1),args=(expo_thres,),disp=True)) # no threshold
    order_dict["exponenital_thr"] = num_evaluations
    num_evaluations += 1
if USE_BAYES == True:
    parameters.append(optimize.fmin(compute_dice_error,(0.1, 1),args=(bayes,),disp=True)) # no threshold
    order_dict["bayes"] = num_evaluations
    num_evaluations += 1
if USE_BAYES_POW == True:
    parameters.append(optimize.fmin(compute_dice_error,(0.6, 0.1),args=(bayes_pow,),disp=True)) # no threshold
    order_dict["bayes_pow"] = num_evaluations
    num_evaluations += 1
if USE_BAYES_THRESHOLD == True:
    parameters.append(optimize.fmin(compute_dice_error,(0.1,0.1,0.1),args=(bayes_thres,),disp=True)) # no threshold
    order_dict["bayes_thr"] = num_evaluations
    num_evaluations += 1
if USE_W_THRESHOLD == True:
    x = optimize.fmin(compute_dice_error,(0.1, 0.1), args=(w_thres,), disp=True)
    order_dict["w_thr"] = num_evaluations
    parameters.append(optimize.fmin(compute_dice_error,(0.1, 0.1), args=(w_thres,), disp=True))
    num_evaluations += 1
