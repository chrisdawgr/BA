import helperfunctions as h
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
  if (Lx != 0):
    orien = math.atan(float(Ly)/float(Lx))
    orien = math.degrees(orien)
    if (orien < 0):
      return(360 + orien)
    return(orien)
  return(0)

def sift_orientation(I, points, window_size):
  """
  Input : image I, interest points, size of window
  Output: assigns an orientation to the interest points
  """

  # point with orientation; [[y, x],[o1, o2, o3]]
  pwo = []
  i = 0
  max_point_size_y = len(I) - (window_size + 2)
  min_point_size_y = window_size
  max_point_size_x = len(I[0]) - (window_size + 5)
  min_point_size_x = window_size 
  hold = numpy.zeros([1,36])

  # length of outer list
  o = 0
  for p in points:
    if ((p[0] <= max_point_size_y and p[0] >= min_point_size_y)) and \
        (p[1] <= max_point_size_x and p[1] >= min_point_size_x):
      o += 1

  # size of all usable points o
  bins = numpy.zeros([o, 1, 36])
  gauss_window = h.gauss(window_size + 2, 1.5)

  for p in points:
    # if a point is too close to the border of the image, it is discarded
    #print(i)
    if ((p[0] <= max_point_size_y and p[0] >= min_point_size_y)) and \
        (p[1] <= max_point_size_x and p[1] >= min_point_size_x):

        ip_window = h.create_window(I, p, window_size + 2) 
        ip_window = numpy.multiply(ip_window, gauss_window)

        # creates bin for each point
        for y in range(0, window_size):
          for x in range(0, window_size):

            magnitude_p = magnitude(ip_window, [y + 1, x + 1])
            orientation_p = orientation(ip_window, [y + 1, x + 1])
            bins[i][0][math.floor(orientation_p /10.0)] += magnitude_p
            hold[0][math.floor(orientation_p /10.0)] += 1

            #print(str([y, x]) + " " + "magnitude: " + str(magnitude_p) \
            #    + " " + "orientation: " + str(orientation_p))

        i += 1
  h.points_to_txt(bins, "bins_for_orientation.txt")
  print(hold)

  
I = cv2.imread('erimitage2.jpg', 0)
points = h.txt_to_points('interest_points.txt')
sift_orientation(I, points, 19)
