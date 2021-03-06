from __future__ import division
import cv2
import numpy
import math
import scipy
import scipy.ndimage
from scipy import misc
from scipy import linalg as LA
import helperfunctions as h

# 3 6400 = 1372
# 1 200000000 =  1393 
# 2 8400 = 1372

def harris(Iname, k, thresh,flag):
  """
  Input : An image, value k and threshold
  Output: Finds interest points using -
          the feature detecor method "Harris corner detector"
  """
  if flag == 1:
    method = "Harris"
    print "Using Harris method with a threshold of:", thresh, "recommended: 200000000"
  if flag == 2:
    method = "HoShiThomas"
    print "Using HoShiThomas method with a threshold of:", thresh, "recommended: 8400"
  if flag == 3:
    method = "Noble"
    print "Using noble method with a threshold of:", thresh, "recommended: 6400"
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
  Ixy = scipy.ndimage.filters.gaussian_filter(Ixy,sigma = 0.7)
  Ixx = scipy.ndimage.filters.gaussian_filter(Ixx,sigma = 0.7)
  Iyy = scipy.ndimage.filters.gaussian_filter(Iyy,sigma = 0.7)
  harrisExtremaPoints = []
  hoShiThomasExtremaPoints = []
  nobleExtremaPoints = []
  
  # Calculate R = Det -k * TR^2:
  for y in range(0, len(I_bw) ):
    for x in range(0, len(I_bw[0])):
      a = Ixx[y][x]
      b = Iyy[y][x]
      c = Ixy[y][x]
      # Harris method:
      if flag == 1:
        C[y][x] = (a*b - c**2) - k * (a + b)**2
      # HoShiThomas method:  
      if flag == 2:
        M = numpy.array([[a, c],[c, b]])
        e_vals, e_vecs = LA.eig(M)
        e_vals = e_vals.real
        C[y][x] = min(e_vals)
      # Noble method:
      if flag == 3:
        epsilon = 0.2
        M = numpy.array([[a, c],[c, b]])
        det = (a*b)-(c*c)
        trace = a + b
        C[y][x] = det/(trace+epsilon)      
  
  # Threshold values to perform edge hysteresis:
  for y in range(3, len(C)):
    for x in range(3, len(C[0])):
      if flag == 1:
        if (C[y][x] > thresh):
          harrisExtremaPoints.append([y,x])
          I[y][x] = [0,0,255]
      if flag == 2:
        if (C[y][x] > thresh):
          hoShiThomasExtremaPoints.append([y,x])
          I[y][x] = [0,0,255]
      if flag == 3:
        if (C[y][x] > thresh):
          nobleExtremaPoints.append([y,x])
          I[y][x] = [0,0,255]

  if flag == 1:
    h.points_to_txt(harrisExtremaPoints, ("harrisExtremaPoints.txt"), "\n")
    print "Found",len(harrisExtremaPoints), "interest points. threshold:",thresh
  if flag == 2:
    h.points_to_txt(hoShiThomasExtremaPoints, "hoShiThomasExtremaPoints.txt", "\n")
    print "Found",len(hoShiThomasExtremaPoints), "interest points. threshold:",thresh
  if flag == 3:
    h.points_to_txt(nobleExtremaPoints, "nobleExtremaPoints.txt", "\n")
    print "Found",len(nobleExtremaPoints), "interest points. threshold:",thresh
  cv2.imwrite(method+str(thresh)+'.jpg', I)

def test_corner_methods(filename,thresh1,thresh2,thresh3):
  harris(filename,0.04,thresh1,1)
  harris(filename,0,thresh2,2)
  harris(filename,0,thresh3,3)

test_corner_methods('erimitage.jpg', 200000000, 8400, 6400)
