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

def find_max(dog1, dog2, dog3, y, x, princip_cur):
  """
  TODO: Point must be x percent larger than nearest.
  Determines if the given point(y,x) is a maximum or minimum point
  among it's 26 neighbours, in the scale above and below it.
  """

  dog11 = h.create_window(dog1, [y, x], 3)
  dog22 = h.create_window(dog2, [y, x], 3)
  dog33 = h.create_window(dog3, [y, x], 3)
  point = dog22[1][1]
  # Create array of neighbouring points 
  dog_points = numpy.array([dog11, dog22, dog33])
  dog_points = dog_points.reshape(-1)            # make 1-dimensional
  dog_points = numpy.delete(dog_points, 13)      # CT: Delete point [1][1]
  
  i = 0
  maxi = 0
  mini = 0
  while(maxi == 0 or mini == 0):
    if (i == 26):
      if(accu_and_edge(dog11, dog22, dog33, y, x, 0, princip_cur) == 1):
        return 1
      return 0
    if dog_points[i] >= point:
      maxi = 1
    if dog_points[i] <= point:
      mini = 1
    i += 1
  return 0


def accu_and_edge(dog1, dog2, dog3, y, x, sigma, princip_cur):
  point = dog2[1][1]
  # Create array of neighbouring points 
  Dxx = (dog2[1][2] + dog2[1][0] - 2 * dog2[1][1]) * 1.0 / 255
  Dyy = (dog2[2][1] + dog2[0][1] - 2 * dog2[1][1]) * 1.0 / 255   
  Dss = (dog3[1][1] + dog1[1][1] - 2 * dog2[1][1]) * 1.0 / 255
  Dxy = (dog2[2][2] - dog2[2][0] - dog2[0][2] + dog2[0][0]) * 0.25 / 255
  Dxs = (dog3[1][2] - dog3[1][0] - dog1[1][2] + dog1[1][0]) * 0.5 / 255 
  Dys = (dog3[2][1] - dog3[0][1] - dog1[2][1] + dog1[0][1]) * 0.5 / 255  
  H = numpy.matrix([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]]) 
  det = float(numpy.linalg.det(H))


  DXX = (dog2[1][2] + dog2[1][0] - 2 * dog2[1][1]) * 1.0 
  DYY = (dog2[2][1] + dog2[0][1] - 2 * dog2[1][1]) * 1.0
  DSS = (dog3[1][1] + dog1[1][1] - 2 * dog2[1][1]) * 1.0
  DXY = (dog2[2][2] - dog2[2][0] - dog2[0][2] + dog2[0][0]) * 0.25 
  DXS = (dog3[1][2] - dog3[1][0] - dog1[1][2] + dog1[1][0]) * 0.5 
  DYS = (dog3[2][1] - dog3[0][1] - dog1[2][1] + dog1[0][1]) * 0.5
  H2 = numpy.matrix([[DXX, DXY, DXS], [DXY, DYY, DYS], [DXS, DYS, DSS]]) 
  det2 = float(numpy.linalg.det(H2))



  if (det > 0):
    Dx = (dog2[1][2] - dog2[1][0]) * 0.5 / 255
    Dy = (dog2[2][1] - dog2[2][1]) * 0.5 / 255
    Ds = (dog3[1][1] - dog1[1][1]) * 0.5 / 255
    DX = numpy.matrix([[Dx], [Dy], [Ds]])
    tr = float(DXX) + float(DYY) + float(DSS)
    r = float(princip_cur)
    xhat = numpy.linalg.inv(H) * DX
    #print(mag_xhat)
    #print(x,y,sigma)

    if (xhat[0] < 0.5 or xhat[1] < 0.5 or xhat[2] < 0.5):
      Dxhat = point + (1/2.0) * DX.transpose() * xhat
      if((abs(Dxhat) > 0.03) and (tr**2/det2 < (r + 1)**2 / r)):
        return 1
      print(tr**2/det2)
      #print(Dxhat)
      #print(xhat)
      #print("rejected because dxhat or  xhat")
      return 0


def SIFT(filename, r_mag):
  """
  Returns the interest points found
  """
  s = 3
  k = 2 ** (1.0 / s)
  I = cv2.imread(filename)
  #I1 = misc.imresize(I, 50, 'bilinear')
  #I2 = misc.imresize(I1, 50, 'bilinear')
  #I3 = misc.imresize(I2, 50, 'bilinear')
  I_bw = cv2.imread(filename, 0)
  dim = I_bw.shape
  height = dim[0]
  length = dim[1]
  #height1 = len(I1)
  #length1 = len(I1[1])
  #height2 = len(I2)
  #length2 = len(I2[1])    
  #height3 = len(I3)
  #length3 = len(I3[1])
  
  sigma1 = [math.sqrt(0.5), math.sqrt(1), math.sqrt(2), math.sqrt(4),
           math.sqrt(8), math.sqrt(16), math.sqrt(32), math.sqrt(64),
           math.sqrt(128), math.sqrt(256), math.sqrt(512)]
  sigma1 = numpy.array([1.3, 1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4)])

  o1sc = [
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma1[0]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma1[1]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma1[2]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma1[3]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma1[4])
  ]

  """
  o2sc = [
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[2]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[3]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[4]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[5]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[6])]
  o3sc = [
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[4]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[5]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[6]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[7]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[8])
  ]
  o4sc = [
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[6]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[7]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[8]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[9]),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = sigma[10])
  ]
  """
  # Calculate difference of gaussian images.
  DoG_scale1 = []
  for i in range(0, 4):
    DoG_scale1.append(h.matrix_substraction(o1sc[i + 1], o1sc[i]))
  
  # Consctruct difference of gaussian pyramids
  # Scale 1:
  dog1 = DoG_scale1[0]
  dog2 = DoG_scale1[1]
  dog3 = DoG_scale1[2]
  dog4 = DoG_scale1[3]
  # Scale 1/2
  """
  dog6 = DoG_scale2[0]
  dog7 = DoG_scale2[1]
  dog8 = DoG_scale2[2]
  dog9 = DoG_scale2[3]
  #scale 1/4
  dog11 = DoG_scale3[0]
  dog12 = DoG_scale3[1]
  dog13 = DoG_scale3[2]
  dog14 = DoG_scale3[3]
  # Scale 1/8
  dog16 = DoG_scale4[0]
  dog17 = DoG_scale4[1]
  dog18 = DoG_scale4[2]
  dog19 = DoG_scale4[3]
  """
  
  # Initialize arrays for keypoints
  DoG_extrema_points_1_1 = []
  DoG_extrema_points_1_2 = []
  DoG_extrema_points_2_1 = []
  DoG_extrema_points_2_2 = []
  DoG_extrema_points_3_1 = []
  DoG_extrema_points_3_2 = []
  DoG_extrema_points_4_1 = []
  DoG_extrema_points_4_2 = []
   
  print("Caluclating extrema, performing accurate keypoint localization and edge supression")
  print("Finding points for scale 1")
  for y in range(3, height - 3):
    for x in range(3, length - 3):
      if (find_max(dog1, dog2, dog3, y, x, r_mag) == 1):
        DoG_extrema_points_1_1.append([y,x])
      if (find_max(dog2, dog3, dog4, y, x, r_mag) == 1):
        DoG_extrema_points_1_2.append([y,x])
  """
  print "Finding points for scale 2"
  for y in range(3, height1 - 3):
    for x in range(3, length1 - 3):
      if (find_max(dog6, dog7, dog8, y, x) == 1):
        DoG_extrema_points_2_1.append([x,y])
      if (find_max(dog7, dog8, dog9, y, x) == 1):
        DoG_extrema_points_2_2.append([x,y])

  print "Finding points for scale 3"
  for y in range(3, height2 - 3):
    for x in range(3, length2 - 3):
      if (find_max(dog11, dog12, dog13, y, x) == 1):
        DoG_extrema_points_3_1.append([x,y])
      if (find_max(dog12, dog13, dog14, y, x) == 1):
        DoG_extrema_points_3_2.append([x,y])

  print "Finding points for scale 4"
  for y in range(3, height3 - 3):
    for x in range(3, length3 - 3):
      if (find_max(dog16, dog17, dog18, y, x) == 1):
        DoG_extrema_points_4_1.append([x,y])
      if (find_max(dog17, dog18, dog19, y, x) == 1):
        DoG_extrema_points_4_2.append([x,y])
  """
  # Scale 1:
  dogn1 =  numpy.array(DoG_extrema_points_1_1)
  dogn2 =  numpy.array(DoG_extrema_points_1_2)
  result = numpy.vstack([dogn1, dogn2]) # set removes dublicates.
  print(result)
  if (len(result) > 3):
    print("ishere")
    print(result)
    h.points_to_txt(result, "interest_points.txt", "\n")
    h.color_pic(I, result, filename[:-4] + "-sift-"+ "r-" + str(r_mag) + ".jpg")
  
  # scale 2:
  #dogn3 =  numpy.array(DoG_extrema_points_2_1)
  #dogn4 =  numpy.array(DoG_extrema_points_2_2)
  #result = set(numpy.vstack([dogn3, dogn4])) # set removes dublicates.
  #h.points_to_txt(result, "interest_points_2.txt", "\n")
  #h.color_pic(I, result, filename[:-4] + "-sift_2-"+ "r-" + str(r_mag) + ".jpg")

  # scale 3:
  #dogn5 =  numpy.array(DoG_extrema_points_3_1)
  #dogn6 =  numpy.array(DoG_extrema_points_3_2)
  #result = set(numpy.vstack([dogn5, dogn6])) # set removes dublicates.
  #h.points_to_txt(result, "interest_points_3.txt", "\n")
  #h.color_pic(I, result, filename[:-4] + "-sift_3-"+ "r-" + str(r_mag) + ".jpg")

  # scale 4:
  #dogn7 =  numpy.array(DoG_extrema_points_4_1)
  #dogn8 =  numpy.array(DoG_extrema_points_4_2)
  #result = set(numpy.vstack([dogn7, dogn8])) # set removes dublicates.
  #h.points_to_txt(result, "interest_points_2.txt", "\n")
  #h.color_pic(I, result, filename[:-4] + "-sift_4-"+ "r-" + str(r_mag) + ".jpg")

def test_SIFT(filename, r, increment, iterations):
  for i in range(0, iterations):
    print(filename, r + (i * increment))
    SIFT(filename, r + (i * increment))

test_SIFT('erimitage2.jpg', 10, 1, 10)
