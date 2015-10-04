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

def getaverage(points):
  return (reduce(lambda x, y: x + y, points) / len(points))

def find_max(dog1, dog2, dog3, y, x):
  """
  TODO: Point must be x percent larger than nearest.
  Determines if the given point(y,x) is a maximum or minimum point
  among it's 26 neighbours, in the scale above and below it.
  """
  point = dog2[1][1] # Shouldnt this come after create window ??

  dog1 = h.create_window(dog1, [y, x], 3)
  dog2 = h.create_window(dog2, [y, x], 3)
  dog3 = h.create_window(dog3, [y, x], 3)
  
  # Create array of neighbouring points 
  dog_points = numpy.array([dog1, dog2, dog3])
  dog_points = dog_points.reshape(-1)            # make 1-dimensional
  dog_points = numpy.delete(dog_points, 13)      # CT: Delete point [1][1]
  scndMin = min(dog_points)
  scndMax = max(dog_points)
  scndAvg = getaverage(dog_points)
  #average = getaverage(dog_points)
  """
  if (point > scndMax) or (point < scndMin):
    return 1
  else:
    return 0 
  """
  if (point > scndMax) or (point < scndMin):
    return 1
  else:
    return 0
  """
  #dx = (dog2[y][x+1] - dog2[x][y-1])*0.5 /255
  dx = (dog2[1][2] - dog2[1][0])*0.5 /255
  #dy = (dog2[y+1][x]- dog2[y-1][x]) * 0.5 / 255
  dy = (dog2[2][1]- dog2[0][1]) * 0.5 / 255
  #ds = (dog3[y][x]- dog1[y][x])*0.5/255 
  ds = (dog3[1][1]- dog1[1][1])*0.5/255 
  #dxx = (dog2[y][x+1] + dog2[y][x-1] - 2 * dog2[y][x]) * 1.0 / 255
  dxx = (dog2[1][2] + dog2[1][0] - 2 * dog2[1][1]) * 1.0 / 255
  #dyy = (dog2[y+1][x] + dog2[y-1][x] - 2 * dog2[y][x]) * 1.0 / 255   
  dyy = (dog2[2][1] + dog2[0][1] - 2 * dog2[1][1]) * 1.0 / 255   
  #dss = (dog3[y][x] + dog1[y][x] - 2 *dog2[y][x]) * 1.0 / 255
  dss = (dog3[1][1] + dog1[1][1] - 2 *dog2[1][1]) * 1.0 / 255
  #dxy = (dog2[y+1][x+1] - dog2[y+1][x-1] - dog2[y-1][x+1] + dog2[y-1][x-1]) * 0.25 / 255 
  dxy = (dog2[2][2] - dog2[2][0] - dog2[0][2] + dog2[0][0]) * 0.25 / 255 
  #dxs = (dog3[y][x+1] - dog3[y][x-1] - dog1[y][x+1] + dog1[y][x-1]) * 0.25 / 255
  dxs = (dog3[1][2] - dog3[1][0] - dog1[1][2] + dog1[1][0]) * 0.25 / 255
  #dys = (dog3[y+1][x] - dog3[y-1][x] - dog1[y+1][x] + dog1[y-1][x]) * 0.25 / 255  
  dys = (dog3[2][1] - dog3[0][1] - dog1[2][1] + dog1[0][1]) * 0.25 / 255  

  dD = numpy.matrix([[dx], [dy], [ds]])
  H = numpy.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
  x_hat = numpy.linalg.lstsq(H, dD)[0]
  D_x_hat = dog2[y][x] + 0.5 * numpy.dot(dD.transpose(), x_hat)

  r = 10.0
  if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (numpy.absolute(x_hat[0]) < 0.5) and (numpy.absolute(x_hat[1]) < 0.5) and (numpy.absolute(x_hat[2]) < 0.5) and (numpy.absolute(D_x_hat) > 0.03):
    return 1
  """
  """
  i = 0
  maxi = 0
  mini = 0
  for i in range(0,25):
    if dog_points[i] > point:
      maxi = 1
    if dog_points[i] < point:
      mini = 1  
    if (maxi == 1 and mini == 1):
      return 0
  return 0
  """

def eliminating_edge_responses(I, points, r):
  """
  """
  result = []
  tr2det = []
  first_points = [] # List of interest points
  
  # appends ALL extremum points for a given scale, to first_points
  for i in range(0, len(points)):
    for j in points[i]:
      first_points.append(map(int, j))
 
  # For each interest point in picture perform edge-elimination:
  for i in range(0, len(first_points)):
    D = h.create_window(I, first_points[i], 3)

    # calculation of the derivaties of D in x, y, xx, yy, yx direction
    #mask_dxx = numpy.array([[0, 0,  0], \
    #                        [1, -2, 1], \
    #                        [0, 0,  0]])

    #dx = (dog2[y][x+1] - dog2[x][y-1])*0.5 /255
    dx = (D[1][2] - D[1][0])*0.5 /255
    #dy = (dog2[y+1][x]- dog2[y-1][x]) * 0.5 / 255
    dy = (D[2][1]- D[0][1]) * 0.5 / 255
    #ds = (dog3[y][x]- dog1[y][x])*0.5/255 
    #ds = (dog3[1][1]- dog1[1][1])*0.5/255 
    #dxx = (dog2[y][x+1] + dog2[y][x-1] - 2 * dog2[y][x]) * 1.0 / 255
    dxx = (D[1][2] + D[1][0] - 2 * D[1][1]) * 1.0 / 255
    #dyy = (dog2[y+1][x] + dog2[y-1][x] - 2 * dog2[y][x]) * 1.0 / 255   
    dyy = (D[2][1] + D[0][1] - 2 * D[1][1]) * 1.0 / 255   
    #dss = (dog3[y][x] + dog1[y][x] - 2 *dog2[y][x]) * 1.0 / 255
    #dss = (dog3[1][1] + dog1[1][1] - 2 *dog2[1][1]) * 1.0 / 255
    #dxy = (dog2[y+1][x+1] - dog2[y+1][x-1] - dog2[y-1][x+1] + dog2[y-1][x-1]) * 0.25 / 255 
    dxy = (D[2][2] - D[2][0] - D[0][2] + D[0][0]) * 0.25 / 255 
    #dxs = (dog3[y][x+1] - dog3[y][x-1] - dog1[y][x+1] + dog1[y][x-1]) * 0.25 / 255
    #dxs = (dog3[1][2] - dog3[1][0] - dog1[1][2] + dog1[1][0]) * 0.25 / 255
    #dys = (dog3[y+1][x] - dog3[y-1][x] - dog1[y+1][x] + dog1[y-1][x]) * 0.25 / 255  
    #dys = (dog3[2][1] - dog3[0][1] - dog1[2][1] + dog1[0][1]) * 0.25 / 255  

    #mask_dyy = mask_dxx.transpose()

    #Dxx = numpy.multiply(D, mask_dxx)
    #Dxx = Dxx.sum()

    #Dyy = numpy.multiply(D, mask_dyy)
    #Dyy = Dyy.sum()
    #Dxy = D[2][2] - D[0][2] - D[2][0] + D[0][0]


    # The Hessian matrix
    # NOTE: when Dxy and Dyx are calculated individually, the
    # result on erimitage.jpg is MUCH better
    #D_hessian = numpy.array([[Dxx, Dxy], \
    #                         [Dxy, Dyy]])
    # trace and determinant of the Hessian matrix

    dD = numpy.matrix([[dx], [dy]])
    H = numpy.matrix([[dxx, dxy], [dxy, dyy]])
    x_hat = numpy.linalg.lstsq(H, dD)[0]
    D_x_hat = D[1][1] + 0.5 * numpy.dot(dD.transpose(), x_hat)
    tr = dxx + dyy
    det = numpy.linalg.det(H)
    # (float(tr)**2 / float(det) < float((r+1)**2) / float(r))
    if (det > 0):
      if ((((dxx + dyy) ** 2) * r) < (dxx * dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (numpy.absolute(x_hat[0]) < 0.5) and (numpy.absolute(x_hat[1]) < 0.5) and (numpy.absolute(D_x_hat) > 0.03):
        result.append(first_points[i])
        tr2det.append(float(tr)**2 / float(det))
  return(result, tr2det)

    # if the determinant is 0, the calculation trace^2/det is invalid,
    # thus an initial check is needed. 
    # The second if, is the check tr^2/det < (r+1)^2/r.
    # in SIFT, the r value is 10
    #if (det > 0):
      #if (float(tr)**2 / float(det) < float((r+1)**2) / float(r)):
        #result.append(first_points[i])
        #tr2det.append(float(tr)**2 / float(det))
  #return(result, tr2det)

def half_image(I):
  """
  Returns an image 1/2 size of the inputted
  """
  return cv2.resize(I, (0,0), fx=0.5, fy=0.5)

# NOTE: SIFT calls the "eliminating_edge_responses function, and shows
# final picture
def SIFT(filename, r_mag):
  """
  Returns the interest points found
  """
  s = 3
  k = 2 ** (1.0 / s)
  I = cv2.imread(filename)
  #I1 = half_image(I)
  #I2 = half_image(I1)
  #I3 = half_image(I2)
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
  
  #sigma1 = [math.sqrt(0.5), math.sqrt(1), math.sqrt(2), math.sqrt(4),
  #         math.sqrt(8), math.sqrt(16), math.sqrt(32), math.sqrt(64),
  #         math.sqrt(128), math.sqrt(256), math.sqrt(512)]
  sigma1 = numpy.array([1.3, 1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4)])

  #o1sctest = list(numpy.zeros((I.shape[0], I.shape[1], 6)))

  #for i in range(0, 6):
    #o1sctest[i] = scipy.ndimage.filters.gaussian_filter(I, sigma = sigma1[i])

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

  # Append for differnt scales of image

  DoG_scale1 = []
  for i in range(0, 4):
    DoG_scale1.append(h.matrix_substraction(o1sc[i + 1], o1sc[i]))

  dog1 = DoG_scale1[0]
  dog2 = DoG_scale1[1]
  dog3 = DoG_scale1[2]
  dog4 = DoG_scale1[3]

    
  """
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
  """
   
  DoG_extrema_points_1_1 = []
  DoG_extrema_points_1_2 = []
  DoG_extrema_points_2 = []
  DoG_extrema_points_3 = []
  DoG_extrema_points_4 = []
   
  for y in range(3, height - 3):
    for x in range(3, length - 3):
      if (find_max(dog1, dog2, dog3, y, x) == 1):
        I[y][x] = [0,0,255]
        DoG_extrema_points_1_1.append([y,x])

      if (find_max(dog2, dog3, dog4, y, x) == 1):
        I[y][x] = [0,0,255]
        DoG_extrema_points_1_2.append([y,x])

  #cv2.imshow('image', I)
  #cv2.waitKey(0)

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

  # cv2.imwrite('erimitage2.jpg',  I)
  vals = [DoG_extrema_points_1_1, DoG_extrema_points_1_2]
  print "eliminating edge responses and performing accurate keypoint localization"
  # eliminating edge responses
  [result1, tr2det1] = eliminating_edge_responses(dog2, \
                    [DoG_extrema_points_1_1], r_mag) 
  [result2, tr2det2] = eliminating_edge_responses(dog3, \
                    [DoG_extrema_points_1_2], r_mag)
  (result1)
  print len(result2)
  result = numpy.concatenate((result1, result2), axis=0)
  tr2det = numpy.concatenate((tr2det1, tr2det2), axis=0)
  totxt = numpy.vstack([result.transpose(),tr2det]).transpose()
  h.points_to_txt(totxt, "interest_points.txt", "\n")

  color_pic(I, result, filename[:-4] + "-sift-"+ "r-" + str(r_mag) + ".jpg")

# input(I, points, name) - points are a list of [y, x] vals, name is optional,
# but should be a string
def color_pic(*arg):
  I = arg[0]
  points = arg[1]

  if (len(arg) >= 2):
    for p in points:
      I[p[0]][p[1]] = [0,0,255]

  if (len(arg) == 3):
    name = arg[2]
    cv2.imwrite(name, I)


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
    D = numpy.zeros([window_size, window_size])
    half_window_size = int(window_size/2)


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
def test_SIFT(filename, r, increment, iterations):
  for i in range(0, iterations):
    SIFT(filename, r + (i * increment))
    print(i)
test_SIFT('erimitage2.jpg', 0.1, 0.1, 15)

#SIFT('erimitage.jpg', 10)
