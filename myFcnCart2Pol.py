# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 19:37:16 2016

@author: Debarghya
"""
import numpy as np
import math
def myFcnCart2Pol(cartImage, radius, angle):
    polarImage = np.uint8(np.zeros([radius,angle]))
    sz = cartImage.shape
    for rad in range(radius-1):
        for theta in range(angle-1):
            xIdx = np.uint32(sz[0]/2 + (rad*sz[0]/(2*(radius-1)))*math.cos((2*math.pi*theta/(angle-1))-math.pi))
            yIdx = np.uint32(sz[1]/2 + (rad*sz[1]/(2*(radius-1)))*math.sin((2*math.pi*theta/(angle-1))-math.pi))
            polarImage[rad,theta] = cartImage[xIdx,yIdx]
    return polarImage