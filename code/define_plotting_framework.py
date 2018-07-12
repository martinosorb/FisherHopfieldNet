import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../code')

from IPython.core.pylabtools import figsize
figsize(14, 7)
from pylab import *
rcParams['figure.subplot.wspace'] = 0.8
rcParams['figure.subplot.hspace'] = 0.6
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['font.size'] = 16
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.frameon'] = False

from hopfieldNetwork import hopfieldNet
from solverFile import solverClass
import numpy as np
import copy
from scipy.special import expit
import matplotlib.pyplot as plt
