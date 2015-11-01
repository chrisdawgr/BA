import math
import cv2
import numpy as np
import helperfunctions as h2


# if (i == 0) = dx, else dy
def create_haar_window(window_size, j):
  """
  input: size of the window, i=0 = dx, i=1 = dy
  output: a haarwindow, of size:  i + (i % 2)
  """
  i = round(j)
  window_size = int(round(window_size))
  round_size = (window_size + window_size % 2)
  haar_window = np.zeros([round_size, round_size])
  half_len = round_size / 2
  if (i == 0):
    for y in range(0, round_size):
      for x in range(0, round_size):
        if (x + 1 > half_len):
          haar_window[y, x] = 1
        else:
          haar_window[y, x] = -1
  
  else:
   for y in range(0, round_size):
        for x in range(0, round_size):
          if (y + 1 > half_len):
            haar_window[y, x] = 1
          else:
            haar_window[y, x] = 0

  return(haar_window)


def haar_size(s):
  return(int(round(s * 2) + (round(2 * s) % 2)))

def round_four(a):
  """
  input: size of the number to round
  output: rounds to nearest 5
  """

  mod_res = a % 4
  if(mod_res > 2):
    return(int(a + (4 - mod_res)))
  else:
    return(int(a - mod_res))


def surf_descriptor(I_name, keypoints):

  """
  input: [y, x, s], name of picture
  output: Descriptor
  """
  I_bw = cv2.imread(I_name, 0)
  I_int = cv2.integral(I_bw)
  filtered_points = []
  keypoints_area = []
  keypoint_descriptor = []
  final_keypoints = []

  # Removing keypoints if the keypoint collecting region is outside the image
  for p in keypoints:
    hs = haar_size(p[2])
    max_size_y = len(I_bw) - 10
    min_size_x = min_size_y = 20 * p[2] - hs
    max_size_x = len(I_bw[0])
    if ((round_four(20 * p[2] + p[0] + hs * 20 + 2) < max_size_y) and (min_size_y < p[0]  ) and \
        (round_four(20 * p[2] + p[1] + hs * 20 + 2) < max_size_x) and (min_size_x < p[1]  )):
      keypoints_area.append([p[0], p[1], p[2]])
      final_keypoints.append([p[0], p[1]])
  
  #keypoint area = [y, x, s]
  for p_a in keypoints_area:
    region_size = round_four(20 * p[2])
    sub_region_size = region_size/4.0
    descriptor = np.zeros([64, 1])
    d_x = np.zeros([20,20])
    d_y = np.zeros([20,20])


    """   
      a   ef  b
        ######
        ######
        ##p###
        ######
        ######
        ######
       c  gh  d 
    the integral square aecg subtracted from fbhd gives the dx-haar-wavelet response 
    """   
    # Calculating the haar wavelet response, in a 20x20 window

    haar_hs = haar_size(p_a[2]) / 2
    incre = sub_region_size/5.0

    y1 = -region_size/2 + int(p_a[0])
    for y in range(0, 20):
      x1 =-region_size/2 + int(p_a[1])
      for x in range(0, 20):
        #print(y1, region_size/2 + int(p_a[1]))
        #print(y1- haar_hs + 1, x1 + haar_hs - 1)
        a = I_int[y1 - haar_hs + 1, x1 + haar_hs - 1]
        b = I_int[y1 - haar_hs + 1, x1 + haar_hs]
        c = I_int[y1 + haar_hs, x1 - haar_hs + 1]
        d = I_int[y1 + haar_hs, x1 + haar_hs]
        e = I_int[y1 + haar_hs - 1, x1]
        f = I_int[y1 - haar_hs + 1, x1 + 1]
        g = I_int[y1 + haar_hs, x1]
        h = I_int[y1 + haar_hs, x1 + 1]
        i = I_int[y1, x1 - haar_hs + 1]
        j = I_int[y1, x1 + haar_hs]
        k = I_int[y1 + 1, x1 - haar_hs + 1]
        l = I_int[y1 + 1, x1 + haar_hs]
        d_x[y, x] = (g + a - c - e) - (d + f - h - b)
        d_y[y, x] = (j + a - i - b) - (d + k - c - l)
        x1 += incre
      y1 += incre

    d_x = np.multiply(h2.gauss(len(d_x), 3.3 * p[2]), d_x)
    d_y = np.multiply(h2.gauss(len(d_x), 3.3 * p[2]), d_y)

    i = 0
    for y in range(0, 4):
      for x in range(0, 4):
        field_dx = np.sum(d_x[y * 5 : (y+1) * 5, x * 5 : (x+1) * 5])
        field_dy = np.sum(d_y[y * 5 : (y+1) * 5, x * 5 : (x+1) * 5])
        field_abs_dx = np.sum(np.abs(d_x[y * 5 : (y+1) * 5, x * 5 : (x+1) * 5]))
        field_abs_dy = np.sum(np.abs(d_y[y * 5 : (y+1) * 5, x * 5 : (x+1) * 5]))

        descriptor[i + 0] = field_dx
        descriptor[i + 1] = field_dy
        descriptor[i + 2] = field_abs_dx
        descriptor[i + 3] = field_abs_dy
        i += 1

    keypoint_descriptor.append(descriptor)

    for k in range(0, len(keypoint_descriptor)):
      keypoint_descriptor[k] = keypoint_descriptor[k] / np.linalg.norm(keypoint_descriptor[k])


  return(keypoint_descriptor, final_keypoints)


#I = cv2.imread("erimitage2.jpg", 0)
#points = surf_descriptor("erimitage2.jpg", p1)
#print(points)
   
"""
def fuck(p):

  I_bw = [100,100]
  filtered_points = []
  keypoints_area = []
  keypoint_descriptor = []

  max_size_x = I_bw[0]
  min_size_x = min_size_y = 20 * p[2]
  max_size_y = I_bw[1]

  if ((p[0] + round_four(20 * p[2]) < max_size_x) and (min_size_x < p[0] ) and \
      (p[1] + round_four(20 * p[2]) < max_size_y) and (min_size_y < p[1] )):
    return True
  return False

print(fuck([25,75,1.2]))
"""



"""
def aa(a):
  d_x_split = np.zeros([16, 4, 4])
  i = 0
  for y in range(0, 4):
    for x in range(0, 4):
      d_x_split[i, :, :] = a[y * 4: (y + 1) * 4, x * 4: (x + 1) * 4]
      i += 1
  return(d_x_split)
  

a = np.zeros([16,16])
a[15, 11] = 15
a[15, 15] = 15
a[15, 7] = 15
a[0, 0] = 15
a[0, 0] = 15
a[0, 0] = 15

aa(a)
print(aa(a))
"""



"""
def kk(a):
    p_a = [[50, 50], np.zeros([10,10])]
    len_p = len(p_a[1]) / 2
    for y in range(-len_p, len_p):
      for x in range(-len_p, len_p):
        p_a[1][y + len_p, x + len_p] = y + x +  p_a[0][0]

    print(p_a)

kk(2)
"""

"""
print(68, round_four(68))
print(56, round_four(56))
print(55, round_four(55))
print(51, round_four(51))
print(44, round_four(41))
"""




"""
    for w in range(0, 4):
      for i in range(0, 13, 4):
        for j in range(0, 4):
          for k in range(0, 4):
            mag_bin[incrementeur][v][j][k] = hold_mag[k + i + j * 16 + w * 64]
            ori = math.floor((360 - hold_ori[k + i + j * 16 + w * 64]) / 45.0) - 1
            #print(hold_ori[k + i + j * 16 + w * 64]) / 45.0))
            ori_bin[incrementeur][v][j][k] = ori
"""
#print(haar_size(2.2))
