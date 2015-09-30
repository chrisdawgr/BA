from __future__ import division
import cv2
import numpy
import math
def harris(Iname, k, thresh):
    I = cv2.imread(Iname)
    I_bw = cv2.imread(Iname, 0)
    Ixy = numpy.empty([len(I_bw),len(I_bw[0])])
    C = numpy.empty([len(I_bw),len(I_bw[0])])

	#mask_dx = numpy.array([-1,0,1]
    mask_dx = numpy.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    mask_dy = numpy.transpose(mask_dx)
    mask_dxx = numpy.transpose(mask_dx)
    mask_dyy = numpy.transpose(mask_dx)
    mask_gauss = numpy.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])
    mask_gauss = numpy.divide(mask_gauss, 273.0)

    # Calculate x and y derivatives of the image:
    Ix = cv2.filter2D(I_bw, -1, mask_dx)
    Iy = cv2.filter2D(I_bw, -1, mask_dy)
    Ixx0 = cv2.filter2D(Ix, -1, mask_dx)
    Ixx = cv2.filter2D(Ixx0, -1, mask_gauss)
    Iyy0 = cv2.filter2D(Iy, -1, mask_dy)
    Iyy = cv2.filter2D(Iyy0, -1, mask_gauss)
    
    # Calculate products of the derivatives
    for y in range(0, len(I_bw) ):
        for x in range(0, len(I_bw[0])):
            Ixy[y][x] = float(Ix[y][x]) * float(Iy[y][x])

    # NOTE: contradicting Harris, calculating Det(I(y,x)), as ab-c1c2, instead
    # of ab-c^2, for possible better results
    Ixy0 = cv2.filter2D(Ix, -1, mask_dy)
    Ixy = cv2.filter2D(Ixy0, -1, mask_gauss)
    Iyx0 = cv2.filter2D(Iy, -1, mask_dx)
    Iyx = cv2.filter2D(Iyx0, -1, mask_gauss)

    # Calculate R
    for y in range(0, len(I_bw) ):
        for x in range(0, len(I_bw[0])):
            a = float(Ixx[y][x])
            b = float(Iyy[y][x])
            c1 = float(Ixy[y][x])
            c2 = float(Iyx[y][x])
            C[y][x] = (a*b - c1* c2) - k * (a + b)**2

	# Set threshold for R
    for y in range(3, len(C)):
        for x in range(3, len(C[0])):
            if (C[y][x] > thresh):
                #print(C[y][x])
                I[y][x] = [0,0,255]

    cv2.imwrite('erimitage-harris.jpg', I)
    cv2.imshow('image', I)
    cv2.waitKey(100000)
harris('erimitage.jpg', 0.04,14920)

def gauss(size, sigma):
    gauss_kernel = numpy.empty([size, size])
    for y in range(0, size):
        for x in range(0, size):
            x1 = x - size/2 
            y1 = y - size/2
            gauss_kernel[y][x] = (1/(2 * math.pi * sigma**2)) * math.e**(-(x1**2 + y1**2)/(2 * sigma**2))
            print(y1, x1)
    return(gauss_kernel)

a = gauss(5,1) * 273
print(a)