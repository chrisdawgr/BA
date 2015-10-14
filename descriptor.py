import helperfunctions as h
import orientat as siftori
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
  return(theta)
 

def decriptor_representation(points_orientations, I):
  window_size = 16
  max_point_size_y = len(I) - (window_size + 2)
  min_point_size_y = window_size
  max_point_size_x = len(I[0]) - (window_size + 5)
  min_point_size_x = window_size 
  gauss_window = h.gauss(window_size, 8)

  final_points = []
  orien_of_bin = []

  len_of_desc = len(points)

  mag_bin = numpy.zeros([len_of_desc, 16, 4, 4])
  ori_bin = numpy.zeros([len_of_desc, 16, 4, 4])
  final_desc = numpy.zeros([len_of_desc, 128])

  incrementeur = 0
  for p in points:

    hold_mag = h.create_window(I, [35,35], window_size)
    hold_ori = h.create_window(I, [35,35], window_size)

    ip_window = h.create_window(I, p, window_size + 2) 

    # creates bin for each point
    for y in range(0, window_size):
      for x in range(0, window_size):
        magnitude_p = magnitude(ip_window, [y + 1, x + 1])
        orientation_p = orientation(ip_window, [y + 1, x + 1])
        hold_mag[y][x] = magnitude_p
        hold_ori[y][x] = orientation_p

    hold_mag = numpy.multiply(hold_mag, gauss_window)
    hold_mag = numpy.reshape(hold_mag, -1)
    hold_ori = numpy.reshape(hold_ori, -1)

    holder = numpy.zeros([4,4])
    v = 0
    for w in range(0, 4):
      for i in range(0, 13, 4):
        for j in range(0, 4):
          for k in range(0, 4):
            mag_bin[incrementeur][v][j][k] = hold_mag[k + i + j * 16 + w * 64]
            ori = math.floor((360 - hold_ori[k + i + j * 16 + w * 64]) / 45.0) - 1
            #print(hold_ori[k + i + j * 16 + w * 64]) / 45.0))
            ori_bin[incrementeur][v][j][k] = ori

        v += 1

    with open('file2.txt','a') as f_handle:
      numpy.savetxt(f_handle, ori_bin[incrementeur], delimiter=" ", fmt="%s")

    #numpy.savetxt('file2.txt', d_bin[incrementeur], delimiter=" ", fmt="%s")
    incrementeur += 1

  mag_val = 0
  ori_val = 0
  #print("\n\n\n")
  for w in range(0, len(points)):
    for i in range(0, 16):
      for j in range(0, 4):
        for k in range(0, 4):
          mag_val = mag_bin[w][i][k][j]
          ori_val = ori_bin[w][i][k][j]
          #print(ori_val, ori_val + 8*i)
          #print("\n\n")
          final_desc[w][8*i + ori_val] += mag_val


  for o in final_desc:
    with open('descriptor.txt','a') as f_handle:
      numpy.savetxt(f_handle, o, delimiter=" ", fmt="%s")



"""
I_bw = cv2.imread('erimitage2.jpg', 0)

points = h.txt_to_3_points_float('interest_P_ori_mag.txt')
#print(points)


decriptor_representation(points, I_bw)
"""
