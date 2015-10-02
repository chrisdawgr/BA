from __future__ import division
import cv2
import numpy
import math
import scipy
import scipy.ndimage
from scipy import misc
from scipy import linalg as LA

def harris(Iname, k, thresh,flag):
  """
  Input : An image, value k and threshold
  Output: Finds interest points using -
          the feature detecor method "Harris corner detector"
  """
  if flag == 1:
    method = "Harris"
    print "Using Harris method with a threshold of:", thresh, "recommended threshold is 200000000"
  if flag == 2:
    method = "HoShiThomas"
    print "Using HoShiThomas method with a threshold of:", thresh, "recommended threshold is 9000"
  # Create empty image holders:
  I = cv2.imread(Iname)
  I_bw = cv2.imread(Iname, 0)
  Ixy = numpy.empty([len(I_bw),len(I_bw[0])])
  Ixx = numpy.empty([len(I_bw),len(I_bw[0])])
  Iyy = numpy.empty([len(I_bw),len(I_bw[0])])
  C = numpy.empty([len(I_bw),len(I_bw[0])])

  # Convolution masks for derivatives and smoothing:
  mask_dx = numpy.array([[-1,0,1],[-1,0,1],[-1,0,1]])
  mask_dy = numpy.transpose(mask_dx)
  mask_gauss = numpy.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],
                            [4,16,26,16,4],[1,4,7,4,1]])
  mask_gauss = numpy.divide(mask_gauss, 273.0)

  # Calculate x and y derivatives:
  Ix = cv2.filter2D(I_bw, -1, mask_dx) # x derivative
  Iy = cv2.filter2D(I_bw, -1, mask_dy) # y derivative

  # Calculate Ix^2, Iy^2 and Ixy:
  for y in range(0, len(I_bw) ):
    for x in range(0, len(I_bw[0])):
      Ixy[y][x] = float(Ix[y][x]) * float(Iy[y][x])
      Ixx[y][x] = float(Ix[y][x]) * float(Ix[y][x])
      Iyy[y][x] = float(Iy[y][x]) * float(Iy[y][x])

  # Accordingly to the notation in Harris' paper:
  # A = Ixx
  # B = Iyy
  # C = Ixy
  # Remove noise, applying a gaussian filter:
  #Ixy = cv2.filter2D(Ixy, -1, mask_gauss)
  #Ixx = cv2.filter2D(Ixx, -1, mask_gauss)
  #Iyy = cv2.filter2D(Iyy, -1, mask_gauss)
  Ixy = scipy.ndimage.filters.gaussian_filter(Ixy,sigma = 0.7)
  Ixx = scipy.ndimage.filters.gaussian_filter(Ixx,sigma = 0.7)
  Iyy = scipy.ndimage.filters.gaussian_filter(Iyy,sigma = 0.7)
  
  # Calculate R = Det -k * TR^2:
  for y in range(0, len(I_bw) ):
    for x in range(0, len(I_bw[0])):
      a = Ixx[y][x]
      b = Iyy[y][x]
      c = Ixy[y][x]
      # Harris method:
      if flag == 1:
        #print "flag = 1"
        C[y][x] = (a*b - c**2) - k * (a + b)**2
      # HoShiThomas method:  
      if flag == 2:
        print "flag = 2"
        M = numpy.array([[a, c],[c, b]])
        e_vals, e_vecs = LA.eig(M)
        e_vals = e_vals.real
        C[y][x] = min(e_vals)
  
  # Threshold values to perform edge hysteresis:
  for y in range(3, len(C)):
    for x in range(3, len(C[0])):
      if flag == 1:
        if (C[y][x] > thresh):
          I[y][x] = [0,0,255]
      if flag == 2:
        if (C[y][x] > thresh):
          I[y][x] = [0,0,255] 
  cv2.imwrite(method+str(thresh)+'.jpg', I)
    #cv2.imshow('image', I)
    #cv2.waitKey(100000)
    #cv2.imshow('image', I)
    #cv2.waitKey(100000)

harris('erimitage.jpg', 0.04, 9000 ,1)