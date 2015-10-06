import cv2
import math
import numpy

def gauss(size, sigma):
  """
  Input : size of window, sigma value
  Output: creates a gauss window of size size
  """
  D = numpy.zeros([size, size])
  gauss_kernel = numpy.zeros([size, size])
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

def create_window(I, point, window_size):
  """
  Input : Image, [y,x], size of window
  Output: Creates a window with size window_size of I, and makes point middle
  """
  D = numpy.empty([window_size, window_size])
  half_w_size = window_size/2
  y_p = point[0]
  x_p = point[1]

  for y in range(0, window_size):
    for x in range(0, window_size):
      D[y][x] = I[y_p - half_w_size + y][x_p - half_w_size + x]

  return(D)


def points_to_txt(points, filename_out, seperate_by):
  """
  input = points, filename to output, how to seperate the lists, eg "\n", "\t", "\n\n" etc
  output = file with filename_out with the points
  """
  file_o = open(filename_out, 'w')

  for i in points:
    file_o.write(str(i[0]) + " " + str(i[1]))
    file_o.write(seperate_by)
  file_o.close()

def txt_to_points(filename):
  result = []
  oo = open(filename, "r")
  points_str = oo.read()
  oo.close()
  points_str = points_str.split()

  for i in range(0, len(points_str), 2):
    result.append([points_str[i], points_str[i + 1]])

  return result


def points_to_txt2(points, filename_out, seperate_by):
  """
  input = points, filename to output, how to seperate the lists, eg "\n", "\t", "\n\n" etc
  output = file with filename_out with the points
  """
  file_o = open(filename_out, 'w')

  for i in points:
    #print(i)
    file_o.write(str(i))
    file_o.write(seperate_by)
  file_o.close()

def txt_to_points(filename):
  result = []
  oo = open(filename, "r")
  points_str = oo.read()
  oo.close()
  points_str = points_str.split()

  for i in range(0, len(points_str), 2):
    result.append([int(points_str[i]), int(points_str[i + 1])])

  return result





def matrix_substraction(m1, m2):
  dim = m1.shape
  height = dim[0]
  length = dim[1]
  mat = numpy.zeros([height, length], dtype='uint8')
  for y in range (0, height):
    for x in range(0, length):
      if (m1[y][x] < m2[y][x]):
        mat[y][x] = 0
      else:
        mat[y][x] = m1[y][x] - m2[y][x]
  return(mat)

      


"""
Testing of the functions
"""


"""
print("this is A:\n")
A = [[I[78][246], I[78][247], I[78][248]], \
     [I[79][246], I[79][247], I[79][248]], \
     [I[80][246], I[80][247], I[80][248]]]
print(numpy.array(A))
print("end of A \n\n")
print("this is D:")
D = create_window(I, [79,247], 5)
print(D)
print("end of D \n\n")
"""


def test_SIFT(filename, r, iterations, increment):
  for i in range(0, iterations):
    sift.SIFT(filename, r + (i * increment))


