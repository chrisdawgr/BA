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

def find_max_new(dog_scale,i,y,x,princip_cur):
  maxpoint = (dog_scale[y, x, i] > 0)
  minpoint = (dog_scale[y, x, i] < 0)
  # Run through 26 neighbours
  for ci in range(-1,2):
    for cy in range(-1,2):
      for cx in range(-1,2):
        if cy == 0 and cx == 0 and ci == 0:
          continue # perform next iteration as we are in orego.
        maxpoint = maxpoint and dog_scale[y,x,i]>dog_scale[y+cy,x+cx,i+ci]
        minpoint = minpoint and dog_scale[y,x,i]<dog_scale[y+cy,x+cx,i+ci]
        # If point lies between max and min, we break
        if not maxpoint and not minpoint:
          return 0
      if not maxpoint and not minpoint:
        return 0
    if not maxpoint and not minpoint:
      return 0
  if maxpoint == True or minpoint == True:
    # Create array of neighbouring points 
    Dxx = (dog_scale[y,x+1,i] + dog_scale[y,x-1,i] - 2 * dog_scale[y,x,i]) * 1.0 / 255
    Dyy = (dog_scale[y+1,x,i] + dog_scale[y-1,x,i] - 2 * dog_scale[y,x,i]) * 1.0 / 255
    Dss = (dog_scale[y,x,i+1] + dog_scale[y,x,i-1] - 2 * dog_scale[y,x,i]) * 1.0 / 255
    Dxy = (dog_scale[y+1,x+1,i] - dog_scale[y+1,x-1,i] - dog_scale[y-1,x+1,i] + dog_scale[y-1,x-1,i]) * 0.25 / 255
    Dxs = (dog_scale[y,x+1,i+1] - dog_scale[y,x-1,i+1] - dog_scale[y,x+1,i-1] + dog_scale[y,x-1,i-1]) * 0.5 / 255 
    Dys = (dog_scale[y+1,x,i+1] - dog_scale[y-1,x,i+1] - dog_scale[y+1,x,i-1] + dog_scale[y-1,x,i-1]) * 0.5 / 255  
    H = numpy.matrix([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]])
    det = float(numpy.linalg.det(H))

    DXX1 = (dog_scale[y,x+1,i] + dog_scale[y,x-1,i] - 2 * dog_scale[y,x,i]) * 1.0 
    DYY1 = (dog_scale[y+1,x,i] + dog_scale[y-1,x,i] - 2 * dog_scale[y,x,i]) * 1.0
    DSS1 = (dog_scale[y,x,i+1] + dog_scale[y,x,i-1] - 2 * dog_scale[y,x,i]) * 1.0
    DXY1 = (dog_scale[y+1,x+1,i] - dog_scale[y+1,x-1,i] - dog_scale[y-1,x+1,i] + dog_scale[y-1,x-1,i]) * 0.25 
    DXS1 = (dog_scale[y,x+1,i+1] - dog_scale[y,x-1,i+1] - dog_scale[y,x+1,i-1] + dog_scale[y,x-1,i-1]) * 0.5 
    DYS1 = (dog_scale[y+1,x,i+1] - dog_scale[y-1,x,i+1] - dog_scale[y+1,x,i-1] + dog_scale[y-1,x,i-1]) * 0.5
    H2 = numpy.matrix([[DXX1, DXY1, DXS1], [DXY1, DYY1, DYS1], [DXS1, DYS1, DSS1]]) 
    det2 = float(numpy.linalg.det(H2))

    if (det > 0) and (det2 != 0):
      Dx = (dog_scale[y,x+1,i] - dog_scale[y,x-1,i]) * 0.5 / 255
      Dy = (dog_scale[y+1,x,i] - dog_scale[y-1,x,i]) * 0.5 / 255
      Ds = (dog_scale[y,x,i+1] - dog_scale[y,x,i-1]) * 0.5 / 255
      DX = numpy.matrix([[Dx], [Dy], [Ds]])
      tr = float(DXX1) + float(DYY1) + float(DSS1)
      r = float(princip_cur)
      xhat = numpy.linalg.inv(H) * DX
      if (abs(xhat[0]) < 0.5 and abs(xhat[1]) < 0.5 and abs(xhat[2]) < 0.5):
        Dxhat = dog_scale[y,x,i] + (1/2.0) * DX.transpose() * xhat # CT old was point, but shouldnt differ
        if((abs(Dxhat) > 1.03) and (tr**2/det2 < (r + 1)**2 / r)):
          return 1
    return 0

def SIFT(filename, r_mag):
  """
  Returns the interest points found
  """
  s = 3
  k = 2 ** (1.0 / s)
  I = cv2.imread(filename)
  I_bw = cv2.imread(filename, 0)
  I1 = misc.imresize(I_bw, 50, 'bilinear').astype(int)
  I2 = misc.imresize(I1, 50, 'bilinear').astype(int)
  I3 = misc.imresize(I2, 50, 'bilinear').astype(int)
  dim = I_bw.shape
  # Sigma value we defined, yields more responses than the one below;
  #sigma1 = [math.sqrt(0.5), math.sqrt(1), math.sqrt(2), math.sqrt(4),
           #math.sqrt(8), math.sqrt(16), math.sqrt(32), math.sqrt(64),
           #math.sqrt(128), math.sqrt(256), math.sqrt(512)]
  # Sigma values for FIRST SCALE ONLY proposed by Lowe, yields less responses:
  sigma1 = numpy.array([1.3, 1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4)])

  print "creating gaussian pyramids.."
  o1sc = numpy.zeros((I_bw.shape[0],I_bw.shape[1],5))
  o2sc = numpy.zeros((I1.shape[0],I1.shape[1],5))
  o3sc = numpy.zeros((I2.shape[0],I2.shape[1],5))
  o4sc = numpy.zeros((I3.shape[0],I3.shape[1],5))

  for i in range(0,5):
    o1sc[:,:,i] = scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma1[i])
    o2sc[:,:,i] = scipy.ndimage.filters.gaussian_filter(I1,sigma = sigma1[i]) #+2
    o3sc[:,:,i] = scipy.ndimage.filters.gaussian_filter(I2,sigma = sigma1[i]) #+4 
    o4sc[:,:,i] = scipy.ndimage.filters.gaussian_filter(I3,sigma = sigma1[i]) # +6
  # Calculate difference of gaussian images.
  print "creating difference of gaussian pyramids.."
  DoG_pictures_scale_1 = numpy.zeros((I_bw.shape[0],I_bw.shape[1],4))
  DoG_pictures_scale_2 = numpy.zeros((I1.shape[0],I1.shape[1],4))
  DoG_pictures_scale_3 = numpy.zeros((I2.shape[0],I2.shape[1],4))
  DoG_pictures_scale_4 = numpy.zeros((I3.shape[0],I3.shape[1],4))

  for i in range(0,4):
    # CT: TRY WITH HELPERFUNCTION MINUS
    DoG_pictures_scale_1[:,:,i] = o1sc[:,:,i+1] - o1sc[:,:,i]
    #DoG_pictures_scale_1[:,:,i] = h.matrix_substraction(o1sc[:,:,i+1],o1sc[:,:,i]) # ct
    DoG_pictures_scale_2[:,:,i] = o2sc[:,:,i+1] - o2sc[:,:,i]
    DoG_pictures_scale_3[:,:,i] = o3sc[:,:,i+1] - o3sc[:,:,i]
    DoG_pictures_scale_4[:,:,i] = o4sc[:,:,i+1] - o4sc[:,:,i]
  #print DoG_pictures_scale_1[:,:,1]
  #cv2.imshow('image',DoG_pictures_scale_1[:,:,0])
  #cv2.waitKey(0)
  
  # Initialize arrays for keypoints
  DoG_extrema_points_1_1 = []
  DoG_extrema_points_1_2 = []
  DoG_extrema_points_2 = []
  DoG_extrema_points_3 = []
  DoG_extrema_points_4 = []
   
  print("Finding points for scale 1")
  for y in range(3, I_bw.shape[0] - 3):
    for x in range(3, I_bw.shape[1] - 3):
      if (find_max_new(DoG_pictures_scale_1,1,y,x, r_mag) == 1):
        DoG_extrema_points_1_1.append([y,x,0])
      if (find_max_new(DoG_pictures_scale_1,2,y,x, r_mag) == 1):
        DoG_extrema_points_1_2.append([y,x,1])
  dogn1 =  numpy.array(DoG_extrema_points_1_1)
  dogn2 = numpy.array(DoG_extrema_points_1_2)
  if (len(dogn1) > 1) and (len(dogn2)>1):
    result = numpy.vstack([dogn1, dogn2])
    #print result
    print "Number of points in first octave: %d" % len(result)
    #h.points_to_txt(result, "interest_points_sc1.txt", "\n")
    h.points_to_txt_3_points(result, "interest_points_sc1.txt", "\n")
    h.color_pic(I, result, filename[:-4] + "-sift_sc1-"+ "r-" + str(r_mag) + ".jpg")
  
  # TODO add scales
  """
  print "Finding points for scale 2"
  for y in range(3, I1.shape[0] - 3):
    for x in range(3,  I1.shape[1] - 3):
      if (find_max_new(DoG_pictures_scale_2,y,x, r_mag) == 1):
        DoG_extrema_points_2.append([y,x,0])
  print len(DoG_extrema_points_2)
  dogn2 =  numpy.array(DoG_extrema_points_2)
  if (len(dogn2) > 1):
    result1 = numpy.vstack([dogn2])
    print "Number of points in second octave: %d" % len(result1)
    h.points_to_txt(result1, "interest_points_sc2.txt", "\n")
    h.color_pic(I1, result1, filename[:-4] + "-sift_sc2-"+ "r-" + str(r_mag) + ".jpg")

  print "Finding points for scale 3"
  for y in range(3, I2.shape[0] - 3):
    for x in range(3,  I2.shape[1] - 3):
      if (find_max_new(DoG_pictures_scale_3,y,x, r_mag) == 1):
        DoG_extrema_points_3.append([y,x,0])
  dogn3 =  numpy.array(DoG_extrema_points_3)
  if (len(dogn3) > 1):
    result2 = numpy.vstack([dogn3])
    print "LENGTH", len(result2)
    print "Number of points in third octave: %d" % len(result2)
    h.points_to_txt(result2, "interest_points_sc3.txt", "\n")
    h.color_pic(I2, result2, filename[:-4] + "-sift_sc3-"+ "r-" + str(r_mag) + ".jpg")  

  print "Finding points for scale 4"
  for y in range(3, I3.shape[0] - 3):
    for x in range(3,  I3.shape[1] - 3):
      if (find_max_new(DoG_pictures_scale_4,y,x, r_mag) == 1):
        DoG_extrema_points_4.append([y,x,0])
  dogn4 =  numpy.array(DoG_extrema_points_4)
  if (len(dogn4) > 1):
    result3 = numpy.vstack([dogn4])
    print "Number of points in fourth octave: %d" % len(result3)
    h.points_to_txt(result3, "interest_points_sc4.txt", "\n")
    h.color_pic(I3, result3, filename[:-4] + "-sift_sc4-"+ "r-" + str(r_mag) + ".jpg")
  """

  return(result)

def test_SIFT(filename, r, increment, iterations):
  for i in range(0, iterations):
    print(filename, r + (i * increment))
    SIFT(filename, r + (i * increment))


test_SIFT('erimitage.jpg', 0.4, 0.1, 1)

