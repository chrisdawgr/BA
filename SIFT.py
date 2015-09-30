#-*- coding: utf-8 -*-
#siftthis.py
#from __future__ import division # CT: Tror vi behøver dette, så når vi skalere et billede skal vi bruge floor/ceil
import scipy
import numpy
import scipy.ndimage
from scipy import misc
import cv2
import math
import csv
from PIL import Image

def create_window(I, point, window_size):
  """
  Create window patch of size window_size
  """
  D = numpy.empty([window_size, window_size])
  half_w_size = window_size/2
  y_p = point[0]
  x_p = point[1]

  for y in range(0, window_size):
    for x in range(0, window_size):
      D[y][x] = I[y_p - half_w_size + y][x_p - half_w_size + x]

  return(D)


#currently assuming, a window size of 3x3
#NOT working properly, and NOT used in SIFT function
def accurate_keypoint_localization(I, vals, window_size):
  final_mat = []
  vals = csv.reader(open('out2.npy', 'r'), delimiter=' ')
  valz = [row for row in vals]
  vals = valz
  first_vals = []
  height = len(I)
  length = len(I[0])   # CT: len(I[0]) ?

  for i in range(0, len(vals)):
    first_vals.append(map(int, vals[i]))

  holder = []
  for i in range(0, len(first_vals)):
    if (first_vals[i][0] > height - 3 or first_vals[i][0] < 3 or \
        first_vals[i][1] > length - 3 or first_vals[i][1] < 3):
      holder.append([first_vals[i][0], first_vals[i][1]])
  first_vals = holder


  first_vals = numpy.array(first_vals)


  for i in range(0, len(first_vals)):

    y_coor = first_vals[i][0] 
    x_coor = first_vals[i][1]
    D = numpy.empty([window_size, window_size])
    half_window_size = int(window_size/2)

    for y in range(0, window_size):
      y1 = y_coor - y + half_window_size
      for x in range(0, window_size):
        x1 = x_coor - x + half_window_size
        #print(y_coor - y + half_window_size, x_coor - x + half_window_size)
        D[y][x] = I[x1][y1]     # CT I[y1][x1] ? 

    #print(D)
    #print("\n")
        

    D_p0_p0 = D[window_size/2 + 0][window_size/2 + 0]
    D_p0_m1 = D[window_size/2 + 0][window_size/2 - 1]
    D_p0_p1 = D[window_size/2 + 0][window_size/2 + 1]
    D_m1_p0 = D[window_size/2 - 1][window_size/2 + 0]
    D_p1_p0 = D[window_size/2 + 1][window_size/2 + 0]

    D_p1_p1 = D[window_size/2 + 1][window_size/2 + 1]
    D_p1_m1 = D[window_size/2 + 1][window_size/2 - 1]
    D_m1_p1 = D[window_size/2 - 1][window_size/2 + 1]
    D_m1_m1 = D[window_size/2 - 1][window_size/2 - 1]

    Dx = float(D_p0_p1 - D_p0_m1) / 2.0
    Dy = float(D_p1_p0 - D_m1_p0) / 2.0
    Dxx = float(D_p0_p1 - 2.0 * D_p0_p0 + D_p0_m1)
    Dyy = float(D_p1_p0 - 2.0 * D_p0_p0 + D_m1_p0)
    Dyx = float(D_m1_m1 + D_p1_p1 - D_m1_p1 - D_p1_m1) / 4.0
    #Dxy = (- D_-1_-1 - D_+1_+1 + D_-1_+1 + D_+1_-1)/4

    D_hessian = numpy.array([[Dxx, Dyx], \
                             [Dyx, Dyy]])


    if (numpy.linalg.det(D_hessian) != 0):
      xhat = - numpy.dot(numpy.linalg.inv(D_hessian),
                numpy.array([[Dx],[Dy]]))
      holder1 = [round(xhat[0]), round(xhat[1])]

      #Check if coordinates are right - might be reversed (x,y)/(y,x)

      D_xhat = D_p0_p0 + (1/2) * numpy.dot(numpy.array([Dy, Dx]), xhat)
      #print(D_xhat)

      if(D_xhat > 10.0):
        final_mat.append(holder1)


      """
      for i in range (0, len(vals)):
        if (abs(D_xhat) > 0.03):
          a.remove(i)
          
      """
  #print(final_mat)





def find_max(dog1, dog2, dog3, y, x):
  """
  Determines if the given point(y,x) is a maximum or minimum point
  among it's 26 neighbours, in the scale above and below it.
  """
  point = dog2[1][1]

  dog1 = create_window(dog1, [y, x], 3)
  dog2 = create_window(dog2, [y, x], 3)
  dog3 = create_window(dog3, [y, x], 3)
  
  # Create array of neighbouring points 
  dog_points = numpy.array([dog1, dog2, dog3])
  dog_points = dog_points.reshape(-1)            # make 1-dimensional
  dog_points = numpy.delete(dog_points, 13)      # CT: Delete point [1][1]

  i = 0
  maxi = 0
  mini = 0
  while(maxi == 0 or mini == 0):
    if dog_points[i] > point:
      maxi = 1
    if dog_points[i] < point:
      mini = 1
    i += 1
    if (i == 26):
      return 1
  return 0


def eliminating_edge_responses(I, vals, window_size, r):
  """
  """
  result = []
  first_vals = [] # List of interest points
  
  # appends ALL extremum points for a given scale, to first_vals
  for i in range(0, len(vals)):
    for j in vals[i]:
      first_vals.append(map(int, j))  # CT: to convert to int? why not use vals directly
                                      # Maybe to convert to 2d array? Looks flat in next paragraph

 
  # For each interest point:
  for i in range(0, len(first_vals)):
    y_coor = first_vals[i][0]        # CT: not used
    x_coor = first_vals[i][1]        # -|-
    D = numpy.empty([window_size, window_size])  # Create 3x3 D
    half_window_size = int(window_size/2)        # CT: Not used


    # D is a window with dimensions 3x3
    D = create_window(I, first_vals[i], 3)

    # calculation of the derivaties of D in x, y, xx, yy, yx direction
    Dx = float(D[1][0] - D[1][2]) / 2.0            # CT: where do calculations come frome
    Dy = float(D[0][1] - D[2][1]) / 2.0
    Dxx = float(D[1][0] - 2.0 * D[1][1] + D[1][2]) # Dxx
    Dyy = float(D[0][1] - 2.0 * D[1][1] + D[2][1]) # Dyy

    # TODO probally not correct
    # dD/dxdy - not sure if the calculation is correct
    Dxy = float(D[2][0] + D[0][2] - D[0][0] - D[2][2]) / 4.0 # CT: page 12, shouldnt they be same?
    Dyx = float(D[0][0] + D[2][2] - D[0][2] - D[2][0]) / 4.0 # 

    # The Hessian matrix
    # Dyx Dxy might be interchanged
    # NOTE: when Dxy and Dyx are calculated individually, the
    # result on erimitage.jpg is MUCH better
    D_hessian = numpy.array([[Dxx, Dxy], \
                             [Dyx, Dyy]]) # CT: shouldnt they be same left to right diagonal?

    # trace and determinant of the Hessian matrix
    tr = Dxx + Dyy
    det = numpy.linalg.det(D_hessian)
    
    # if the determinant is 0, the calculation trace^2/det is invalid,
    # thus an initial check is needed. 
    # The second if, is the check tr^2/det < (r+1)^2/r.
    # in SIFT, the r value is 10
    if (det != 0):
      if ((tr**2) / det < (float(r**2) / float(r))):
        result.append(first_vals[i])

  return(result)


def half_image(I):
  """
  Returns an image 1/2 size of the inputted
  """
  return cv2.resize(I, (0,0), fx=0.5, fy=0.5)

def gauss(size, sigma):
  gauss_kernel = numpy.empty([size, size])
  for y in range(0, size):
    for x in range(0, size):
      half = int(size/2)
      x1 = x - half
      y1 = y - half
      frac = (1.0/(2.0 * math.pi * sigma**2)) 
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
  return(gauss_kernel)


# NOTE: SIFT calls the "eliminating_edge_responses function, and shows
# final picture
def SIFT(Iname, k, sigma):
  """
  Returns the interest points found
  """
  I = cv2.imread(Iname)
  I1 = half_image(I)
  I2 = half_image(I1)
  I3 = half_image(I2)
  I_bw = cv2.imread(Iname, 0)
  dim = I_bw.shape
  height = dim[0]
  length = dim[1]
  height1 = len(I1)
  length1 = len(I1[1])
  height2 = len(I2)
  length2 = len(I2[1])    
  height3 = len(I3)
  length3 = len(I3[1])
  

  o1sc = [
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(0.5)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(1)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(2)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(4)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(8))
  ]

  o2sc = [
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(2)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(4)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(8)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(16)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(32))]

  o3sc = [
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(8)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(16)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(32)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(64)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(128))
  ]

  o4sc = [
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(32)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(64)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(128)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(256)),
  scipy.ndimage.filters.gaussian_filter(I_bw,sigma = math.sqrt(512))
  ]

  DoG_scale1 = []
  DoG_scale2 = []
  DoG_scale3 = []
  DoG_scale4 = []

  for i in range(0, 4):
    DoG_scale1.append(o1sc[i + 1] - o1sc[i])
    DoG_scale2.append((half_image(o2sc[i+1]))-(half_image(o2sc[i])))
    DoG_scale3.append((half_image(half_image(o3sc[i+1])))-\
                     (half_image(half_image(o3sc[i]))))
    DoG_scale4.append((half_image(half_image(half_image(o4sc[i+1]))))-\
                     ((half_image(half_image(half_image(o4sc[i]))))))

  dog1 = DoG_scale1[0]
  dog2 = DoG_scale1[1]
  dog3 = DoG_scale1[2]
  dog4 = DoG_scale1[3]

  dog6 = DoG_scale2[0]
  dog7 = DoG_scale2[1]
  dog8 = DoG_scale2[2]
  dog9 = DoG_scale2[3]
    
  dog11 = DoG_scale3[0]
  dog12 = DoG_scale3[1]
  dog13 = DoG_scale3[2]
  dog14 = DoG_scale3[3]

  dog16 = DoG_scale4[0]
  dog17 = DoG_scale4[1]
  dog18 = DoG_scale4[2]
  dog19 = DoG_scale4[3]
   
  DoG_extrema_points_1_1 = []
  DoG_extrema_points_1_2 = []
  DoG_extrema_points_2 = []
  DoG_extrema_points_3 = []
  DoG_extrema_points_4 = []
   
  for y in range(3, height - 3):
    for x in range(3, length - 3):
      if (find_max(dog1, dog2, dog3, y, x) == 1):
        #I[y][x] = [0,0,255]
        DoG_extrema_points_1_1.append([y,x])

      if (find_max(dog2, dog3, dog4, y, x) == 1):
        #I[y][x] = [0,0,255]
        DoG_extrema_points_1_2.append([y,x])

  """
  for y in range(3, height1 - 3):
    for x in range(3, length1 - 3):
      if (find_max(dog6, dog7, dog8, y, x) == 1):
        I1[y][x] = [0,0,255]
        DoG_extrema_points_2.append([x,y])

      if (find_max(dog7, dog8, dog9, y, x) == 1):
        I1[y][x] = [0,0,255]
        DoG_extrema_points_2.append([x,y])

  for y in range(3, height2 - 3):
    for x in range(3, length2 - 3):
      if (find_max(dog11, dog12, dog13, y, x) == 1):
        I2[y][x] = [0,0,255]
        DoG_extrema_points_3.append([x,y])

      if (find_max(dog12, dog13, dog14, y, x) == 1):
        I2[y][x] = [0,0,255]
        DoG_extrema_points_3.append([x,y])

  for y in range(3, height3 - 3):
    for x in range(3, length3 - 3):
      if (find_max(dog16, dog17, dog18, y, x) == 1):
        I3[y][x] = [0,0,255]
        DoG_extrema_points_4.append([x,y])

      if (find_max(dog17, dog18, dog19, y, x) == 1):
        I3[y][x] = [0,0,255]
        DoG_extrema_points_4.append([x,y])
  """

  # Writing points to file "out.txt"
  with open('out.txt', 'wb') as f:
    csv.writer(f, delimiter=' ').writerows(DoG_extrema_points_1_2)


  # cv2.imwrite('erimitage2.jpg',  I)
  vals = [DoG_extrema_points_1_1, DoG_extrema_points_1_2]

  # eliminating edge responses
  result1 = eliminating_edge_responses(dog2, [DoG_extrema_points_1_1], 3, 0.2)
  result2 = eliminating_edge_responses(dog3, [DoG_extrema_points_1_2], 3, 0.2)
  result = numpy.concatenate((result1, result2), axis=0)

  color_pic(I, result)

  
  #cv2.imwrite('DoG1.jpg',I1)
  #cv2.imwrite('DoG2.jpg',I2)
  #cv2.imwrite('DoG3.jpg',I3)

  #cv2.imshow('image', I)
  #cv2.waitKey(100000)
  #cv2.imshow('image', testp)
  #cv2.waitKey(100000)


# input(I, points, name) - points are a list of [y, x] vals, name is optional,
# but should be a string
def color_pic(arg):
  I = arg[0]
  vals = arg[1]

  if (len(arg) >= 2):
    for a in vals:
      I[a[0]][a[1]] = [0,0,255]
    cv2.imshow('image', I)
    cv2.waitKey(0)

  if (len(arg) == 3):
    name = arg[2]
    cv2.imwrite(name, I)

SIFT('erimitage2.jpg', 1.5, 2)
