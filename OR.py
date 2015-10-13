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
  #new 1
  bins1 = numpy.zeros([o,1,36])
  bins2 = numpy.zeros([o,1,36])
  bins3 = numpy.zeros([o,1,36])
  bins4 = numpy.zeros([o,1,36])
  bins5 = numpy.zeros([o,1,36])
  bins6 = numpy.zeros([o,1,36])
  bins7 = numpy.zeros([o,1,36])
  bins8 = numpy.zeros([o,1,36])
  bins9 = numpy.zeros([o,1,36])
  bins10 = numpy.zeros([o,1,36])
  bins11 = numpy.zeros([o,1,36])
  bins12 = numpy.zeros([o,1,36])
  bins13 = numpy.zeros([o,1,36])
  bins14 = numpy.zeros([o,1,36])
  bins15 = numpy.zeros([o,1,36])
  bins16 = numpy.zeros([o,1,36])
  #new 2

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
      # new
      W1 = h.create_window(hold_mag, [2,2], 4)
      W2 = h.create_window(hold_mag, [2,6], 4)
      W3 = h.create_window(hold_mag, [2,10], 4)
      W4 = h.create_window(hold_mag, [2,14], 4)
      W5 = h.create_window(hold_mag, [6,2], 4)
      W6 = h.create_window(hold_mag, [6,6], 4)
      W7 = h.create_window(hold_mag, [6,10], 4)
      W8 = h.create_window(hold_mag, [6,14], 4)
      W9 = h.create_window(hold_mag, [10,2], 4)
      W10 = h.create_window(hold_mag, [10,6], 4)
      W11 = h.create_window(hold_mag, [10,10], 4)
      W12 = h.create_window(hold_mag, [10,14], 4)
      W13 = h.create_window(hold_mag, [14,2], 4)
      W14 = h.create_window(hold_mag, [14,6], 4)
      W15 = h.create_window(hold_mag, [14,10], 4)
      W16 = h.create_window(hold_mag, [14,14], 4)
      O1 = h.create_window(hold_ori, [2,2], 4)
      O2 = h.create_window(hold_ori, [2,6], 4)
      O3 = h.create_window(hold_ori, [2,10], 4)
      O4 = h.create_window(hold_ori, [2,14], 4)
      O5 = h.create_window(hold_ori, [6,2], 4)
      O6 = h.create_window(hold_ori, [6,6], 4)
      O7 = h.create_window(hold_ori, [6,10], 4)
      O8 = h.create_window(hold_ori, [6,14], 4)
      O9 = h.create_window(hold_ori, [10,2], 4)
      O10 = h.create_window(hold_ori, [10,6], 4)
      O11 = h.create_window(hold_ori, [10,10], 4)
      O12 = h.create_window(hold_ori, [10,14], 4)
      O13 = h.create_window(hold_ori, [14,2], 4)
      O14 = h.create_window(hold_ori, [14,6], 4)
      O15 = h.create_window(hold_ori, [14,10], 4)
      O16 = h.create_window(hold_ori, [14,14], 4)
      # new
      hold_mag = numpy.reshape(hold_mag, -1)
      hold_ori = numpy.reshape(hold_ori, -1)
      # new
      W1 = numpy.reshape(W1, -1)
      W2 = numpy.reshape(W2, -1)
      W3 = numpy.reshape(W3, -1)
      W4 = numpy.reshape(W4, -1)
      W5 = numpy.reshape(W5, -1)            
      W6 = numpy.reshape(W6, -1)
      W7 = numpy.reshape(W7, -1)
      W8 = numpy.reshape(W8, -1)
      W9 = numpy.reshape(W9, -1)
      W10 = numpy.reshape(W10, -1)
      W11 = numpy.reshape(W11, -1)
      W12 = numpy.reshape(W12, -1)
      W13 = numpy.reshape(W13, -1)
      W14 = numpy.reshape(W14, -1)
      W15 = numpy.reshape(W15, -1)
      W16 = numpy.reshape(W16, -1)
      O1 = numpy.reshape(O1, -1)
      O2 = numpy.reshape(O2, -1)
      O3 = numpy.reshape(O3, -1)
      O4 = numpy.reshape(O4, -1)
      O5 = numpy.reshape(O5, -1)            
      O6 = numpy.reshape(O6, -1)
      O7 = numpy.reshape(O7, -1)
      O8 = numpy.reshape(O8, -1)
      O9 = numpy.reshape(O9, -1)
      O10 = numpy.reshape(O10, -1)
      O11 = numpy.reshape(O11, -1)
      O12 = numpy.reshape(O12, -1)
      O13 = numpy.reshape(O13, -1)
      O14 = numpy.reshape(O14, -1)
      O15 = numpy.reshape(O15, -1)
      O16 = numpy.reshape(O16, -1)
      for j in range(0,len(O1)):
        bins1[i][0][O1[j]] += W1[j]
        bins2[i][0][O2[j]] += W2[j]
        bins3[i][0][O3[j]] += W3[j]
        bins4[i][0][O4[j]] += W4[j]
        bins5[i][0][O5[j]] += W5[j]
        bins6[i][0][O6[j]] += W6[j]
        bins7[i][0][O7[j]] += W7[j]
        bins8[i][0][O8[j]] += W8[j]
        bins9[i][0][O9[j]] += W9[j]
        bins10[i][0][O10[j]] += W10[j]
        bins11[i][0][O11[j]] += W11[j]
        bins12[i][0][O12[j]] += W12[j]
        bins13[i][0][O13[j]] += W13[j]
        bins14[i][0][O14[j]] += W14[j]
        bins15[i][0][O15[j]] += W15[j]
        bins16[i][0][O16[j]] += W16[j]
      # new
      
      for j in range(0, len(hold_ori)):
        bins[i][0][hold_ori[j]] += hold_mag[j]

      hold_mag = h.create_window(I[0], [35,35], window_size)
      hold_ori = h.create_window(I[0], [35,35], window_size)

      bin_i = bins[i][0]
      # new
      bin_i_1 = bins1[i][0]
      bin_i_2 = bins2[i][0]
      bin_i_3 = bins3[i][0]
      bin_i_4 = bins4[i][0]
      bin_i_5 = bins5[i][0]
      bin_i_6 = bins6[i][0]
      bin_i_7 = bins7[i][0]
      bin_i_8 = bins8[i][0]
      bin_i_9 = bins9[i][0]
      bin_i_10 = bins9[i][0]
      bin_i_11 = bins11[i][0]
      bin_i_12 = bins12[i][0]
      bin_i_13 = bins13[i][0]
      bin_i_14 = bins14[i][0]
      bin_i_15 = bins15[i][0]
      bin_i_16 = bins16[i][0]
      max_val1 = max(bin_i_1)
      max_index1 = [k for k, j in enumerate(bin_i_1) if j == max_val1]
      max_val2 = max(bin_i_2)
      max_index2 = [k for k, j in enumerate(bin_i_2) if j == max_val2]
      max_val3 = max(bin_i_3)
      max_index3 = [k for k, j in enumerate(bin_i_3) if j == max_val3]
      max_val4 = max(bin_i_4)
      max_index4 = [k for k, j in enumerate(bin_i_4) if j == max_val4]
      max_val5 = max(bin_i_5)
      max_index5 = [k for k, j in enumerate(bin_i_5) if j == max_val5]
      max_val6 = max(bin_i_6)
      max_index6 = [k for k, j in enumerate(bin_i_6) if j == max_val6]
      max_val7 = max(bin_i_7)
      max_index7 = [k for k, j in enumerate(bin_i_7) if j == max_val7]
      max_val8 = max(bin_i_8)
      max_index8 = [k for k, j in enumerate(bin_i_8) if j == max_val8]
      max_val9 = max(bin_i_9)
      max_index9 = [k for k, j in enumerate(bin_i_9) if j == max_val9]
      max_val10 = max(bin_i_10)
      max_index10 = [k for k, j in enumerate(bin_i_10) if j == max_val10]
      max_val11 = max(bin_i_11)
      max_index11 = [k for k, j in enumerate(bin_i_11) if j == max_val11]
      max_val12 = max(bin_i_12)
      max_index12 = [k for k, j in enumerate(bin_i_12) if j == max_val12]
      max_val13 = max(bin_i_13)
      max_index13 = [k for k, j in enumerate(bin_i_13) if j == max_val13]
      max_val14 = max(bin_i_14)
      max_index14 = [k for k, j in enumerate(bin_i_14) if j == max_val14]
      max_val15 = max(bin_i_15)
      max_index15 = [k for k, j in enumerate(bin_i_15) if j == max_val15]
      max_val16 = max(bin_i_16)
      max_index16 = [k for k, j in enumerate(bin_i_16) if j == max_val16]
      # new

        
      # index of max element in bin
      max_index = max(bin_i)
      max_index = [k for k, j in enumerate(bin_i) if j == max_index]
        
      # finds points within 80% of interest point

      holder_bin = []
      holder_bin.append(max_index5[0])
      max_val = bin_i[max_index5]

      for j in range(0, 35):
        if (bin_i[j] >= max_val * 0.8 and j != max_index5[0]):
          holder_bin.append(j)
      orien_of_bin.append(holder_bin)
    i += 1


  o = open('orientss.txt', 'w')
  for i in orien_of_bin:
    o.write(str(i) + "\n")
  o.close()
  #print(orien_of_bin)
  #print(len(bins))



I_bw = cv2.imread('erimitage2.jpg', 0)

"""
print(orientation22(I, 1, 0))
print(orientation22(I, 1, 1))
print(orientation22(I, 0, 1))
print(orientation22(I, -1, 1))
print(orientation22(I, -1, 0))
print(orientation22(I, -1, -1))
print(orientation22(I, 0, -1))
print(orientation22(I, 1, -1))
"""
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

sift_orientation(I, points,16)