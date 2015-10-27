import cv2
import math
import numpy as np

def gauss(size, sigma):
  """
  Creates Gusssian kernel
  """
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  half = int(size/2)
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      frac = (1.0/(2.0 * math.pi * sigma**2)) 
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(gauss_kernel))
  return(gauss_kernel)


def gaussdx(size, sigma):
  """
  Creates a kernel with the first derivative of gauss
  """
  #NOTE: Guys at NVidia suggests a gausskernel of size 3*sigma
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  half = int(size/2)
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      gauss_calc = (-((x1/(2.0*math.pi*sigma**4.0)*(math.exp(-((x1**2.0+y1**2.0)/(2.0*sigma**2.0)))))))
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(gauss_kernel))
  return(gauss_kernel)

def gaussdy(size, sigma):
  return np.transpose((gaussdx(size,sigma)))

def gauss2x(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  half = int(size/2)
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      frac = -1*((x1**2-sigma**2)/(2.0 * math.pi * sigma**4.0))
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(gauss_kernel))
  return(gauss_kernel)

def gauss2y(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  half = int(size/2)
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      frac = -1*((y1**2-sigma**2)/(2.0 * math.pi * sigma**6.0))
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(gauss_kernel))
  return(gauss_kernel)

def gauss2xy(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  half = int(size/2)
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      first = ((x1*y1)/2.0*math.pi**6.0)
      second = math.exp(-((x1**2.0+y1**2.0)/(2.0*sigma**2.0)))
      gauss_calc = first*second
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(gauss_kernel))
  print sum(sum(gauss_kernel))
  return(gauss_kernel)


def create_window(I, point, window_size):
  """
  Input : Image, [y,x], size of window
  Output: Creates a window with size window_size of I, and makes point middle
  """
  D = np.empty([window_size, window_size])
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

def points_to_txt_3_points(points, filename_out, seperate_by):
  """
  input = points, filename to output, how to seperate the lists, eg "\n", "\t", "\n\n" etc
  output = file with filename_out with the points
  """
  file_o = open(filename_out, 'w')

  for i in points:
    file_o.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]))
    file_o.write(seperate_by)
  file_o.close()


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


def txt_to_3_points(filename):
  result = []
  oo = open(filename, "r")
  points_str = oo.read()
  oo.close()
  points_str = points_str.split()
  for i in range(0, len(points_str), 3):
    result.append([int(points_str[i]), int(points_str[i + 1]), int(points_str[i + 2])])
  return result


def txt_to_3_points_float(filename):
  result = []
  oo = open(filename, "r")
  points_str = oo.read()
  oo.close()
  points_str = points_str.split()
  for i in range(0, len(points_str), 3):
    result.append([int(points_str[i]), int(points_str[i + 1]), float(points_str[i + 2])])
  return result


def matrix_substraction(m1, m2):
  dim = m1.shape
  height = dim[0]
  length = dim[1]
  mat = np.zeros([height, length], dtype='uint8')
  for y in range (0, height):
    for x in range(0, length):
      if (m1[y][x] < m2[y][x]):
        mat[y][x] = 0
      else:
        mat[y][x] = m1[y][x] - m2[y][x]
  return(mat)

def color_pic(*arg):
  """ 
  input: I, points
  output: pic
  """
  I = arg[0]
  points = arg[1]

  if (len(arg) >= 2):
    for p in points:
      I[p[0]][p[1]] = [0,0,255]
    cv2.imshow('image', I)
    cv2.waitKey(0)

  if (len(arg) == 3):
    name = arg[2]
    cv2.imwrite(name, I)
