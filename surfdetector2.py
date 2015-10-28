from __future__ import division
import scipy
import numpy
import scipy.ndimage
from scipy import misc
import cv2
import math
import csv
from PIL import Image
import helperfunctions as h
from scipy import signal
from scipy import misc
from scipy import ndimage

# Dedicated to my dear, my muse, my Andreas

def getGauss(size,i):
  sigma = 1.2
  if i == 0:
    return h.gauss(9,sigma)
  if i == 1:
    return h.gauss(15,sigma)
  if i == 2:
    return h.gauss(21,sigma)
  if i == 3:
    return h.gauss(27,sigma)
  if i == 4:
    return h.gauss(39,sigma)
  if i == 5:
    return h.gauss(51,sigma)
  if i == 6:
    return h.gauss(75,sigma)
  if i == 7:
    return h.gauss(99,sigma)


def accurate_keypoint(deriv):
  # deriv [dxx,dyy,dxy]
  Dxx = deriv[0]
  Dyy = deriv[1]
  Dxy = deriv[2]
  Dx  = deriv[3]
  Dy  = deriv[4]
  #print (Dxy)
  #Dxs = deriv3[3]-deriv1[3]*0.5 #0.5 This value is much larger
  #Dys = deriv3[4]-deriv1[4]*0.5 #0.5 This value is much more negative
  #Dss = deriv3[5]-deriv1[5] * 0.5 # this value is 0
  #Dx = deriv2[3] * 0.5  # 0.5
  #Dy = deriv2[4] * 0.5  # 0.5
  #Ds = deriv2[5] * 0.5 # 0.5 # is zero
  #H = numpy.matrix([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]])
  #print (H)
  H = numpy.matrix([[Dxx, Dxy], [Dxy, Dyy]])
  det = float(numpy.linalg.det(H))
  #DX = numpy.matrix([[Dx], [Dy], [Ds]])
  DX = numpy.matrix([[Dx], [Dy]])
  #print (Dxy)
  #print (det)
  if det != 0:
    xhat = numpy.linalg.inv(H) * DX
    #print ("xhat:",xhat[0],"\t",xhat[1])
    #print xhat
    if (abs(xhat[0]) < 10 and abs(xhat[1]) < 10):
      #print "passed xhat"
      #print (abs(xhat[0]))
      Dxhat = ((1/2.0) * DX.transpose() * xhat) # Missing point
      #print (Dxhat)
      print Dxhat
      if((abs(Dxhat) > 1.03)):
        return 1
      print "rejected by dxhat"
    print ("rejected xhat")
    return 0
  return 0

def find_max_new(scale,i,y,x,):
  maxpoint = (scale[y, x, i] > 0)
  minpoint = (scale[y, x, i] < 0)
  # Run through 26 neighbours
  for ci in range(-1,2):
    for cy in range(-1,2):
      for cx in range(-1,2):
        if cy == 0 and cx == 0 and ci == 0:
          continue # perform next iteration as we are in orego.
        maxpoint = maxpoint and scale[y,x,i]>scale[y+cy,x+cx,i+ci]
        minpoint = minpoint and scale[y,x,i]<scale[y+cy,x+cx,i+ci]
        #print scale[y+cy,x+cx,i+ci]
        # If point lies between max and min, we break
        if not maxpoint and not minpoint:
          return 0
      if not maxpoint and not minpoint:
        return 0
    if not maxpoint and not minpoint:
      return 0
  if maxpoint == True or minpoint == True:  
    return 1

def findSurfPoints(filename):
  clear = " " * 50
  I_bw = cv2.imread(filename, 0).astype(float)
  I = cv2.imread(filename)
  sigma = 1.2

  gausspictures = numpy.zeros((I_bw.shape[0],I_bw.shape[1],8))
  
  for i in range (0,7):
    gausspictures[:,:,i] = cv2.filter2D(I_bw,-1, getGauss(1.2,i))

  deriv9 = numpy.zeros((I_bw.shape[0],I_bw.shape[1],5))
  deriv15 = numpy.zeros((I_bw.shape[0],I_bw.shape[1],5))
  deriv21 = numpy.zeros((I_bw.shape[0],I_bw.shape[1],5))
  deriv27 = numpy.zeros((I_bw.shape[0],I_bw.shape[1],5))

  for y in range (10, I_bw.shape[0]-10):
    for x in range (10, I_bw.shape[1]-10):
      deriv9[y,x,0] = gausspictures[y,x+1,0] + gausspictures[y,x-1,0] - 2 * gausspictures[y,x,0] # DXX
      deriv9[y,x,1] = gausspictures[y+1,x,0] + gausspictures[y-1,x,0] - 2 * gausspictures[y,x,0] # DYY
      deriv9[y,x,2] = gausspictures[y+1,x+1,0] - gausspictures[y+1,x-1,0] - gausspictures[y-1,x+1,0] + gausspictures[y-1,x-1,0] # DXY
      deriv9[y,x,3] = gausspictures[y,x+1,0] - gausspictures[y,x-1,0] # DX
      deriv9[y,x,4] = gausspictures[y+1,x,0] - gausspictures[y+1,x,0] # DY
      # TODO Also calculate ds
      
      deriv15[y,x,0] = gausspictures[y,x+1,1] + gausspictures[y,x-1,1] - 2 * gausspictures[y,x,1] # DXX
      deriv15[y,x,1] = gausspictures[y+1,x,1] + gausspictures[y-1,x,1] - 2 * gausspictures[y,x,1] # DYY
      deriv15[y,x,2] = gausspictures[y+1,x+1,1] - gausspictures[y+1,x-1,1] - gausspictures[y-1,x+1,1] + gausspictures[y-1,x-1,1] # DXY
      deriv15[y,x,3] = gausspictures[y,x+1,1] - gausspictures[y,x-1,1]
      deriv15[y,x,4] = gausspictures[y+1,x,1] - gausspictures[y+1,x,1]


      deriv21[y,x,0] = gausspictures[y,x+1,2] + gausspictures[y,x-1,2] - 2 * gausspictures[y,x,2] # DXX
      deriv21[y,x,1] = gausspictures[y+1,x,2] + gausspictures[y-1,x,2] - 2 * gausspictures[y,x,2] # DYY
      deriv21[y,x,2] = gausspictures[y+1,x+1,2] - gausspictures[y+1,x-1,2] - gausspictures[y-1,x+1,2] + gausspictures[y-1,x-1,2] # DXY
      deriv21[y,x,3] = gausspictures[y,x+1,2] - gausspictures[y,x-1,2]
      deriv21[y,x,4] = gausspictures[y+1,x,2] - gausspictures[y+1,x,2]

      deriv27[y,x,0] = gausspictures[y,x+1,3] + gausspictures[y,x-1,3] - 2 * gausspictures[y,x,3] # DXX
      deriv27[y,x,1] = gausspictures[y+1,x,3] + gausspictures[y-1,x,3] - 2 * gausspictures[y,x,3] # DYY
      deriv27[y,x,2] = gausspictures[y+1,x+1,3] - gausspictures[y+1,x-1,3] - gausspictures[y-1,x+1,3] + gausspictures[y-1,x-1,3] # DXY
      deriv27[y,x,3] = gausspictures[y,x+1,3] - gausspictures[y,x-1,3]
      deriv27[y,x,4] = gausspictures[y+1,x,3] - gausspictures[y+1,x,3]

  hessian9  = numpy.zeros((I_bw.shape[0],I_bw.shape[1]))
  hessian15 = numpy.zeros((I_bw.shape[0],I_bw.shape[1]))
  hessian21 = numpy.zeros((I_bw.shape[0],I_bw.shape[1]))
  hessian27 = numpy.zeros((I_bw.shape[0],I_bw.shape[1]))

  for y in range (10, I_bw.shape[0]-10):
    for x in range (10, I_bw.shape[1]-10):
      hessian9[y,x] = (deriv9[y,x,0] * deriv9[y,x,1]) - (0.9*deriv9[y,x,2])**2
      hessian15[y,x] = (deriv15[y,x,0] * deriv15[y,x,1]) - (0.9*deriv15[y,x,2])**2
      hessian21[y,x] = (deriv21[y,x,0] * deriv21[y,x,1]) - (0.9*deriv21[y,x,2])**2
      hessian27[y,x] = (deriv27[y,x,0] * deriv27[y,x,1]) - (0.9*deriv27[y,x,2])**2
  
  scale1hessian = numpy.zeros((I_bw.shape[0],I_bw.shape[1],4))
  scale1hessian[:,:,0] = hessian9
  scale1hessian[:,:,1] = hessian15
  scale1hessian[:,:,2] = hessian21
  scale1hessian[:,:,3] = hessian27


  extrema_points_1_1 = []
  extrema_points_1_2 = []
  
  for y in range(0,I_bw.shape[0]):
    for x in range(0,I_bw.shape[1]):
      Flag = False
      if find_max_new(scale1hessian,1,y,x) == 1 and (accurate_keypoint(deriv15[y,x,:]) == 1):
        extrema_points_1_1.append([y,x,(9/9*1.2)])
        Flag = True
      if Flag == False and find_max_new(scale1hessian,2,y,x) == 1  and (accurate_keypoint(deriv21[y,x,:]) == 1):
        extrema_points_1_2.append([y,x,(15/9*1.2)])
  dogn1 =  numpy.array(extrema_points_1_1)
  dogn2 = numpy.array(extrema_points_1_2)
  if (len(dogn1) > 1) and (len(dogn2)>1):
    result = numpy.vstack([dogn1, dogn2])
    print ("Number of points in first octave: %d" % len(result))
    h.points_to_txt_3_points(result, "SURF_interest_points_o1.txt", "\n")
    h.color_pic(I, result, filename[:-4] + "Surfo1" + ".jpg")


findSurfPoints("erimitage2.jpg")