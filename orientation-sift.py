import helperfunctions as h
import scipy
import scipy.ndimage
from scipy import misc
import cv2
import math
import numpy


def magnitude(I, point):
  """
  Input : Image I, point point
  Output: outputs the magnitude
  """
  w = h.create_window(I, point, 3)
  mag = math.sqrt((w[1][0] - w[1][2])**2 + (w[0][1] - w[2][1])**2)
  return(mag)


def orientation(I, point):
  """
  Input : Image I, point point
  Output: orientation of image
  """
  w = h.create_window(I, point, 3)
  Ly = w[0][1] - w[2][1]
  Lx = w[1][0] - w[1][2]
  theta = 0.5 * math.pi - math.atan2(Lx, Ly)
  theta = math.degrees(theta % (2 * math.pi))
  #print(theta)
  return(theta)
 

def sift_orientation(I, points, window_size):
  """
  Input : image I, interest points, size of window
  Output: assigns an orientation to the interest points
  """

  # point with orientation; [[y, x],[o1, o2, o3]]
  pwo = []
  max_point_size_y = len(I[0]) - (window_size + 2)
  min_point_size_y = window_size
  max_point_size_x = len(I[0][0]) - (window_size + 5)
  min_point_size_x = window_size 
  gauss_window = h.gauss(window_size, 1.5)
  hold_mag = h.create_window(I[0], [35,35], window_size)
  hold_ori = h.create_window(I[0], [35,35], window_size)
  final_points = []
  orien_of_bin = []
  

  # length of outer list
  o = 0
  for p in points:
    if ((p[0] < max_point_size_y and min_point_size_y < p[0]) and \
        (p[1] < max_point_size_x and min_point_size_x < p[1])):
      final_points.append(p)
      o += 1

  # size of all usable points o
  bins = numpy.zeros([o,  1, 36])

  i = 0
  for p in final_points:
    # if a point is too close to the border of the image, it is discarded
    if ((p[0] < max_point_size_y and p[0] > min_point_size_y) and \
        (p[1] < max_point_size_x and p[1] > min_point_size_x)):

      ip_window = h.create_window(I[p[2]], p, window_size + 2) 

      # creates bin for each point
      for y in range(0, window_size):
        for x in range(0, window_size):
          magnitude_p = magnitude(ip_window, [y + 1, x + 1])
          orientation_p = orientation(ip_window, [y + 1, x + 1])
          orientation_p = math.floor(orientation_p/10.0)
          hold_mag[y][x] = magnitude_p
          hold_ori[y][x] = orientation_p

      hold_mag = numpy.multiply(hold_mag, gauss_window)
      hold_mag = numpy.reshape(hold_mag, -1)
      hold_ori = numpy.reshape(hold_ori, -1)

      for j in range(0, len(hold_ori)):
        bins[i][0][hold_ori[j]] += hold_mag[j]

      hold_mag = h.create_window(I[0], [35,35], window_size)
      hold_ori = h.create_window(I[0], [35,35], window_size)

      bin_i = bins[i][0]
        
      # index of max element in bin
      max_index = max(bin_i)
      max_index = [k for k, j in enumerate(bin_i) if j == max_index]
        
      # finds points within 80% of interest point

      holder_bin = []
      holder_bin.append(max_index[0])
      max_val = bin_i[max_index]

      for j in range(0, 35):
        if (bin_i[j] >= max_val * 0.8 and j != max_index[0]):
          holder_bin.append(j)
      orien_of_bin.append(holder_bin)
    i += 1

  new_orien = list(orien_of_bin)
  for i in range(0, len(orien_of_bin)):
    holder = []
    for j in orien_of_bin[i]:
      if (j == 1):
        A = 0
        B = bins[i][0][j] 
        C = bins[i][0][j]
      if (j == 35):
        A = bins[i][0][j-1]
        B = bins[i][0][j]
        C = 0
      else:
        A = bins[i][0][j - 1]
        B = bins[i][0][j]
        C = bins[i][0][j + 1]
      a = A + (C-A)/2.0 -B
      b = (C-A)/2.0
      c = B
      toppoint = -b / (2 * a)
      point = toppoint * 10 + j * 10

      if (point < 0):
        point = 360.0 + point
      holder.append(point)
    new_orien[i] = holder

  o = open('orients.txt', 'w')
  for i in  range(0, len(orien_of_bin)):
    o.write(str(orien_of_bin[i]) + "\t" + str(new_orien[i]) + "\n")
  o.close()
  print([points,new_orien])
  return ([points, new_orien])



I_bw = cv2.imread('erimitage2.jpg', 0)

points = h.txt_to_3_points('interest_points_with_sigma.txt')
#print(points)

k = 2**(1.0/2)
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

I = [o1sc[1], o1sc[2]]

a = sift_orientation(I, points,16)
