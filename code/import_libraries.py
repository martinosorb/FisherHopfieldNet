import time
import random
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
from scipy.special import expit
from scipy.misc import imresize
import sys
sys.path.append('../code')
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
