# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:10:26 2016

@author: Debarghya
"""
import numpy as np
import math
def myFcnPol2Cart(polarImage,row,col):
    cartImage = np.uint8(np.zeros([row,col]))
    sz = polarImage.shape
    for r in range(row-1):
        for c in range(col-1):
            phi = min(sz[1]-1,np.uint32(((math.atan2((c-(col/2)),(r-(row/2)))+math.pi)*sz[1])/(2*math.pi))-0)
            radcap = min(sz[0]-1,np.uint32((math.sqrt((r-(row/2))**2+(c-(col/2))**2)*sz[0])/(row*0.5))-0)
            cartImage[r,c] = polarImage[radcap,phi]
    return cartImage;