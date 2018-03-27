# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:21:15 2015

@author: montoya
"""

import matplotlib.pyplot as plt
import img_util as imgu
import numpy as np
import scipy.io


# Matlab data files are like dictionaries !!

data = scipy.io.loadmat('seizure.mat')

Xtrain = data['Xtrain']
Ytrain = data['Ytrain']