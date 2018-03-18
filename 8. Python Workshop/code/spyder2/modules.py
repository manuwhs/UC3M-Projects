# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

""" Basic import and namespaces """
import math as math
print math.cos(3.14/4)

import numpy as np
print np.cos(3.14/4)

""" Different forms of input"""

import numpy as np
random_number = np.random.uniform(0,1,10)

import numpy.random as random
random_number2 = random.uniform(0,1,10)

from numpy.random import uniform
random_number3 = uniform(0,1,10)


""" Importing our own module"""

import modulo_basico as mb
mb.print_suma(5, 6)
print mb.numero_magico





    