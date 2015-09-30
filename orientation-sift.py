import siftthis as sift
import harristhis as harris
import cv2
import math
import numpy


def sift_orientation(I, points, window_size):
  for p in points:
    y = p[0]
    x = p[1]



def create_window(I, point, window_size):
  D = numpy.empty([window_size, window_size])
  half_w_size = window_size/2
  y_p = point[0]
  x_p = point[1]

  for y in range(0, window_size):
    for x in range(0, window_size):
      D[y][x] = I[y_p - half_w_size + y][x_p - half_w_size + x]

  return(D)

I = cv2.imread('erimitage.jpg', 0)
D = create_window(I, [100,100], 3)



# Test of create_window function
"""
print("this is A:\n")
A = [[I[99][99],  I[99][100],  I[99][101]], \
     [I[100][99], I[100][100], I[100][101]], \
     [I[101][99], I[101][100],  I[101][101]]]
print(numpy.array(A))
print("end of A \n\n")

print("this is D:")
print(D)
print("end of D \n\n")



points = harris.harris('erimitage.jpg', 0.04, 12000)
"""
