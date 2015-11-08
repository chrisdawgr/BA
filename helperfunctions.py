import cv2
import math
import random
import numpy as np

def gauss(size, sigma):
  """
  Creates Gusssian kernel
  """
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  half = int(size/2)
  if (size > 9):
    stepsize = 9.0 / (size + (size/9))
  else:
    stepsize = 9.0 / (size)
  y1 = -4.0
  for y in range(0, size):
    x1 = -4.0
    for x in range(0, size):
      frac = (1.0/(2.0 * math.pi * sigma**2)) 
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(abs(gauss_kernel)))
  return(gauss_kernel)


def gaussdx(size, sigma):
  """
  Creates a kernel with the first derivative of gauss
  """
  #NOTE: Guys at NVidia suggests a gausskernel of size 3*sigma
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  if (size > 9):
    stepsize = 9.0 / (size + (size/9))
  else:
    stepsize = 9.0 / (size)
  half = int(size/2)
  y1 = -4.0
  for y in range(0, size):
    x1 = -4.0
    for x in range(0, size):
      gauss_calc = (((-x1/(2.0*math.pi*sigma**4.0)*(math.exp(-((x1**2.0+y1**2.0)/(2.0*sigma**2.0)))))))
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(abs(gauss_kernel)))
  return(gauss_kernel)

def gaussdy(size, sigma):
  return np.transpose((gaussdx(size,sigma)))

def gauss2x(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  half = int(size/2)
  if (size > 9):
    stepsize = 9.0 / (size + (size/9))
  else:
    stepsize = 9.0 / (size)
  y1 = -4.0
  for y in range(0, size):
    x1 = -4.0
    for x in range(0, size):
      frac = -1*((x1**2-sigma**2)/(2.0 * math.pi * sigma**4.0))
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(abs(gauss_kernel)))
  return(gauss_kernel)

def gauss2y(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / size
  half = int(size/2)
  if (size > 9):
    stepsize = 9.0 / (size + (size/9))
  else:
    stepsize = 9.0 / (size)

  y1 = -4.0
  for y in range(0, size):
    x1 = -4.0
    for x in range(0, size):
      frac = -1*((y1**2-sigma**2)/(2.0 * math.pi * sigma**6.0))
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(np.abs(gauss_kernel)))
  return(gauss_kernel)

def gauss2xy(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  if (size > 9):
    stepsize = 9.0 / (size + (size/9))
  else:
    stepsize = 9.0 / (size)

  half = int(size/2)
  y1 = -4
  for y in range(0, size):
    x1 = -4
    for x in range(0, size):
      first = ((x1*y1)/2.0*math.pi**6.0)
      second = math.exp(-((x1**2.0+y1**2.0)/(2.0*sigma**2.0)))
      gauss_calc = first*second
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(np.abs(gauss_kernel)))
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

def color_scale(scale1):
    if (scale1 > 1.9 and scale1 < 2.1):
      return (0, 0, 255)
    if (scale1 > 2.7 and scale1 < 2.9):
      return (0, 0, 155)
    if (scale1 > 3.5 and scale1 < 3.6):
      return (0, 255, 0)
    if (scale1 > 5.1 and scale1 < 5.3):
      return (0, 155, 0)
    if (scale1 > 6.7 and scale1 < 6.9):
      return (255, 0, 0)
    if (scale1 > 9.9 and scale1 < 10.1):
      return (155, 0, 0)


def drawMatches(I1, kp1, I2, kp2, matches):
  """
  img1,img2 - Grayscale images
  kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
            detection algorithms
  matches - A list of matches of corresponding keypoints through any
            OpenCV keypoint matching algorithm
  """
  print("drawing matches")
  img1 = cv2.imread(I1, 0)
  img2 = cv2.imread(I2, 0)

  # Create a new output image that concatenates the two images together
  # (a.k.a) a montage
  rows1 = len(img1)
  cols1 = len(img1[0])
  rows2 = len(img2)
  cols2 = len(img2[0])

  out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

  # Place the first image to the left
  out[:rows1,:cols1] = np.dstack([img1, img1, img1])

  # Place the next image to the right of it
  out[:rows2,cols1:] = np.dstack([img2, img2, img2])

  # For each pair of points we have between both images
  # draw circles, then connect a line between them
  for mat in range(0, len(kp1)):

    # Get the matching keypoints for each of the images
    # x - columns
    # y - rows
    (y1,x1,scale1) = (kp1[mat])
    (y2,x2,scale2) = (kp2[mat])

    color1 = color_scale(scale1)
    color2 = color_scale(scale2)

    # Draw a small circle at both co-ordinates
    # radius 4
    # colour blue
    # thickness = 1
    cv2.circle(out, (int(x1),int(y1)), 2 * int(scale1), color1, 3)   
    cv2.circle(out, (int(x2)+cols1,int(y2)), 2 * int(scale2), color2, 3)

    # Draw a line in between the two points
    # thickness = 1
    # colour blue

    if (scale1 != scale2):
      cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (125,0,125), 3)
    else:
      cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color1, 1)


  # Show the image
  cv2.imshow('Matched Features', out)
  cv2.imwrite(str(I1) + "-advanced-matching.jpg", out)
  cv2.waitKey(0)
  cv2.destroyWindow('Matched Features')

  # Also return the image if you'd like a copy
  return out

def oneNN(descs1, descs2, p1, p2):
  print("calculating oneNN")
  print(len(descs1), len(p1))
  print(len(descs2), len(p2))

  res_1 = []
  res_2 = []
  res_d_1 = []
  res_d_2 = []
  descs1 = np.array(descs1)
  descs2 = np.array(descs2)

  for i_desc1 in range(0, len(descs1)):
    desc1 = descs1[i_desc1]
    s_dist = float("inf")
    s_dist_index = 0
    for i_desc2 in range(0, len(descs2)):
      desc2 = descs2[i_desc2]
      dist = np.linalg.norm(desc2 - desc1)

      if (dist < s_dist):
        s_dist = dist
        s_dist_index = i_desc2

    res_1.append(p1[i_desc1])
    res_d_1.append(desc1)
    res_2.append(p2[s_dist_index])
    res_d_2.append(descs2[s_dist_index])

  """
  I = cv2.imread("room10.jpg")
  I2 = cv2.imread("room11.jpg")
  for fin in range(0, len(res_1)):
    I[res_1[fin][0], res_1[fin][1]] = (0,0,255)
    I2[res_2[fin][0], res_2[fin][1]] = (0,0,255)
    cv2.circle(I, (res_1[fin][1].astype(int), res_1[fin][0].astype(int)), 10, (0,255,0), 3)
    cv2.circle(I2, (res_2[fin][1].astype(int), res_2[fin][0].astype(int)), 10, (0,255,0), 3)

  cv2.imwrite("zzonenn" + "room10" + ".jpg", I)
  cv2.imwrite("zzonenn" + "room11" + ".jpg", I2)

  """
  #res_1 = np.array(res_1)
  #res_2 = np.array(res_2)
  #res_d_1 = np.array(res_d_1)
  #res_d_2 = np.array(res_d_2)
  return(res_1, res_2, res_d_1, res_d_2)


def oneNN_wdist(descs1, descs2, p1, p2):
  print("calculating oneNN sorted after euclidean distance")
  print(len(descs1), len(p1))
  print(len(descs2), len(p2))

  res_1 = []
  res_2 = []
  res_d_1 = []
  res_d_2 = []
  res_d_d = []
  descs1 = np.array(descs1)
  descs2 = np.array(descs2)

  for i_desc1 in range(0, len(descs1)):
    desc1 = descs1[i_desc1]
    s_dist = float("inf")
    s_dist_index = 0
    for i_desc2 in range(0, len(descs2)):
      desc2 = descs2[i_desc2]
      dist = np.linalg.norm(desc2 - desc1)

      if (dist < s_dist):
        s_dist = dist
        s_dist_index = i_desc2

    res_1.append(p1[i_desc1])
    res_d_1.append(desc1)
    res_2.append(p2[s_dist_index])
    res_d_2.append(descs2[s_dist_index])
    res_d_d.append(s_dist)

  res_d_d_np = np.array(res_d_d)
  res_1_np = np.array(res_1)
  res_2_np = np.array(res_2)
  res_d_1_np = np.array(res_d_1)
  res_d_2_np = np.array(res_d_2)
  index = np.argsort(res_d_d_np)
  print(res_d_d_np[index])
  return(res_1_np[index], res_2_np[index], res_d_1_np[index], res_d_2_np[index])

  """
  I = cv2.imread("room10.jpg")
  I2 = cv2.imread("room11.jpg")
  for fin in range(0, len(res_1)):
    I[res_1[fin][0], res_1[fin][1]] = (0,0,255)
    I2[res_2[fin][0], res_2[fin][1]] = (0,0,255)
    cv2.circle(I, (res_1[fin][1].astype(int), res_1[fin][0].astype(int)), 10, (0,255,0), 3)
    cv2.circle(I2, (res_2[fin][1].astype(int), res_2[fin][0].astype(int)), 10, (0,255,0), 3)

  cv2.imwrite("zzonenn" + "room10" + ".jpg", I)
  cv2.imwrite("zzonenn" + "room11" + ".jpg", I2)

  """
  return(res_1, res_2, res_d_1, res_d_2)


"""
def cut_img(I_name, quadrant):
  img = cv2.imread(I_name)
  (col, row, _) = np.shape(img)
  if (quadrant == 1)
  img = img[0:col/2][
"""
  




def advanced_oneNN(descss1, descss2, pp1, pp2):
  print("calculating advanced oneNN")
  (res_p1, res_p2, res_des1, res_des2) = oneNN(descss1, descss2, pp1, pp2)
  new_res_p1 = []
  new_res_p2 = []


  for point1 in range(0, len(res_p1)):
    fst_shortest_index = point1
    fst_shortest_dist = np.linalg.norm(res_des1[point1] - res_des2[point1])
    scn_shortest_index = 0
    scn_shortest_dist = float("inf")
    
    for point2 in range(0, len(pp2)):
      if (np.all(descss2[point2] != res_des2[point1])):
        dist = np.linalg.norm(res_des1[point1] - descss2[point2])
        #print(dist, fst_shortest_dist)
        if (scn_shortest_dist > dist):
          scn_shortest_dist = dist
          scn_shortest_index = point2

      #else:
      #  print(point2, "these are identical")

    if (fst_shortest_dist / scn_shortest_dist > 0.2):
      #print(fst_shortest_dist/ scn_shortest_dist)
      new_res_p1.append(res_p1[point1])
      new_res_p2.append(res_p2[point1])

  """
  I = cv2.imread("mark-seg-2-1.jpg")
  I2 = cv2.imread("mark-seg-2-2.jpg")
  print("length of the adv_oneNN finall points ", len(new_res_p1))
  #np.save("new_res_p1", new_res_p1)
  #np.save("new_res_p2", new_res_p2)

  cv2.imwrite("zzonennadv" + "one" + ".jpg", I)
  cv2.imwrite("zzonennadv" + "two" + ".jpg", I2)
  """

  print("returned")
  return(new_res_p1, new_res_p2)
