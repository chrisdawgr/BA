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

def accurate_keypoint(point,deriv1, deriv2, deriv3):
  # deriv [dxx,dyy,dxy,dx,dy,gauss]
  Dxx = deriv2[0] * 0.25 /255
  Dyy = deriv2[1] * 0.25/ 255 
  Dxy = deriv2[2] * 0.25 /255 #this value is much larger WRONG
  #print (Dxx)
  Dxs = (deriv3[3]-deriv1[3])*0.5/255 #0.5 This value is much larger
  Dys = (deriv3[4]-deriv1[4])*0.5/255 #0.5 This value is much more negative
  Dss = (-2 * deriv2[5] + deriv3[5] + deriv1[5]) * 0.25/255 # this value is 0
  #print (Dss)
  Dx = deriv2[3] * 0.5/255  # 0.5
  Dy = deriv2[4] * 0.5/255  # 0.5
  Ds = (deriv3[5] - deriv1[5]) * 0.5/255 # 0.5 # is zero
  #H = numpy.matrix([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]])
  #print (H)
  H = numpy.matrix([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]])
  det = float(numpy.linalg.det(H))
  #DX = numpy.matrix([[Dx], [Dy], [Ds]])
  DX = numpy.matrix([[Dx], [Dy], [Ds]])
  #print (Dxy)
  #print (det)
  if det != 0:
    xhat = numpy.linalg.inv(H) * DX
    #print ("xhat:",xhat[0],"\t",xhat[1]) 
    #print (xhat)
    if (abs(xhat[0]) < 0.5 and abs(xhat[1]) < 0.5):# and abs(xhat[2]) < 7):# way too low
      #print (abs(xhat[0]))
      Dxhat = (point + (1/2.0) * DX.transpose() * xhat) #  This is way too big. Missing point
      print(Dxhat)
      #print (10*(abs(Dxhat[0])))
      if(abs(Dxhat) > 1.03):
        #print ("passed dxhat")
        return 1
      #print ("rejected dxhat")
    #print ("rejected xhat")
    return 0
  return 0

def getGauss(size,i):
  sigma = 1.2
  if i == 0:
    return h.gauss2x(size,sigma)
  if i == 1:
    return h.gauss2y(size,sigma)
  if i == 2:
    return h.gauss2xy(size,sigma)
  if i == 3:
    return h.gaussdx(size,sigma/3)
  if i == 4:
    return h.gaussdy(size,sigma/3)
  if i == 5:
    return h.gauss(size,sigma/3)

def findSurfPoints(filename):
  clear = " " * 50
  I_bw = cv2.imread(filename, 0).astype(float)
  I = cv2.imread(filename)
  
  # Initialize gaussian kernel holders
  testfilter = numpy.zeros((5,5,6))
  filter9 = numpy.zeros((9,9,6))
  filter15 = numpy.zeros((15,15,6))
  filter21 = numpy.zeros((21,21,6))
  filter27 = numpy.zeros((27,27,6))
  filter39 = numpy.zeros((39,39,6))
  filter51 = numpy.zeros((51,51,6))
  filter75 = numpy.zeros((75,75,6))
  filter99 = numpy.zeros((99,99,6))

  #print("Process: Calculating Gaussian kernels","\r", end="")
  # Get gaussian kernels [dxx,dyy,dxy,dx,dy,gauss]
  for i in range(0,6):
    filter9[:,:,i] = getGauss(9,i)
    filter15[:,:,i] = getGauss(15,i)
    filter21[:,:,i] = getGauss(21,i)
    filter27[:,:,i] = getGauss(27,i)
    filter39[:,:,i] = getGauss(39,i)
    filter51[:,:,i] = getGauss(51,i)
    filter75[:,:,i] = getGauss(75,i)
    filter99[:,:,i] = getGauss(99,i)
  #print ("\n")
  #print (numpy.sum(filter9[:,:,1]))
  #print (numpy.sum(filter9[:,:,0]))

  # Intitialize convolved image holder
  conv9  = numpy.zeros((I_bw.shape[0], I_bw.shape[1],6))
  conv15 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],6))
  conv21 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],6))
  conv27 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],6))
  conv39 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],6))
  conv51 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],6))
  conv75 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],6))
  conv99 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],6))

  # Each holder has the image corresponding to dxx, dyy, dxy, dx, dy, gauss, respectively
  #print(clear,"\r", end="")
  #print("Process: Convolving image for all derivates","\r", end="")
  for i in range(0,6):
    conv9[:,:,i]  = cv2.filter2D(I_bw, -1, filter9[:,:,i])
    conv15[:,:,i] = cv2.filter2D(I_bw, -1, filter15[:,:,i])
    conv21[:,:,i] = cv2.filter2D(I_bw, -1, filter21[:,:,i])
    conv27[:,:,i] = cv2.filter2D(I_bw, -1, filter27[:,:,i])
    conv39[:,:,i] = cv2.filter2D(I_bw, -1, filter39[:,:,i])
    conv51[:,:,i] = cv2.filter2D(I_bw, -1, filter51[:,:,i])
    conv75[:,:,i] = cv2.filter2D(I_bw, -1, filter75[:,:,i])
    conv99[:,:,i] = cv2.filter2D(I_bw, -1, filter99[:,:,i])
  #print (conv9[:,:,1])
  #print (conv9[:,:,2])
  
  # Initialize holders for determinants of hessian
  o1 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],4))
  o2 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],4))
  o3 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],4))
  o4 = numpy.zeros((I_bw.shape[0], I_bw.shape[1],4))

  # Calculate determinant for each octave
  for y in range(10,I_bw.shape[0]-10):
    for x in range(10,I_bw.shape[1]-10):
      o1[y,x,0] = conv9[y,x,0]*conv9[y,x,1]-((0.9*conv9[y,x,2])**2)
      o1[y,x,1] = conv15[y,x,0]*conv15[y,x,1]-((0.9*conv15[y,x,2])**2)
      o1[y,x,2] = conv21[y,x,0]*conv21[y,x,1]-((0.9*conv21[y,x,2])**2)
      o1[y,x,3] = conv27[y,x,0]*conv27[y,x,1]-((0.9*conv27[y,x,2])**2)

  extrema_points_1_1 = []
  extrema_points_1_2 = []
  #extrema_points_2 = []
  #extrema_points_3 = []
  #extrema_points_4 = [] 
  #print(clear,"\r", end="")

  #print("Process: Finding points for first octave","\r", end="")

  # Perform non maximal supression on determinant of Hessian.
  passedmax, passeddxhat, rejectedxhat = 0,0,0
  passedmax1, passeddxhat1, rejectedxhat1 = 0,0,0

  for y in range(0,I_bw.shape[0]):
    for x in range(0,I_bw.shape[1]):
      Flag = False
      if find_max_new(o1,1,y,x) == 1:
        passedmax += 1
        if (accurate_keypoint(o1[y,x,1],conv9[y,x,:],conv15[y,x,:],conv21[y,x,:]) == 1):
          passeddxhat += 1
          extrema_points_1_1.append([y,x,(9/9*1.2)])
        rejectedxhat += 1
        Flag = True
      if Flag == False and find_max_new(o1,2,y,x) == 1:
        passedmax1 += 1
        if (accurate_keypoint(o1[y,x,2],conv9[y,x,:],conv15[y,x,:],conv21[y,x,:]) == 1):
          passeddxhat1 += 1
          extrema_points_1_2.append([y,x,(15/9*1.2)])
        rejectedxhat1 += 1
  print ("number of passed by dxhat:",passeddxhat, "of",passedmax)
  print (passeddxhat)
  print ("number of passed dxhat:",passeddxhat1, "of",passedmax1)
  print ("total number of rejected by accurate_keypoint",((passedmax+passedmax1)-passeddxhat+passeddxhat1))
  dogn1 =  numpy.array(extrema_points_1_1)
  dogn2 = numpy.array(extrema_points_1_2)
  if (len(dogn1) > 1) and (len(dogn2)>1):
    result = numpy.vstack([dogn1, dogn2])
    print ("Number of points in first octave: %d" % len(result))
    h.points_to_txt_3_points(result, "SURF_interest_points_o1.txt", "\n")
    h.color_pic(I, result, filename[:-4] + "Surfo1" + ".jpg")

            
#findSurfPoints("erimitage2.jpg")


"""

"""
