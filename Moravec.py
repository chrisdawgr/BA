import cv2

def moravec(a):
  I = cv2.imread(a);
  I_bw = cv2.imread(a,0);
  arr = [[0 for x in range(0,len(I_bw[0]))] for y in range(0,len(I_bw))]

  for x in range(3, len(I_bw) - 3):
    for y in range(3, len(I_bw[0]) - 3):
      m1 = moravec_window(I_bw, x, y, -1, 1)
      m2 = moravec_window(I_bw, x, y,  0, 1)
      m3 = moravec_window(I_bw, x, y,  1, 1)
      m4 = moravec_window(I_bw, x, y,  1, 0)
      arr[x][y] = min(m1, m2, m3, m4)

  for x in range(0, len(arr)):
    for y in range(0, len(arr[0])):
      if ((arr[x][y]) > 29000):
        I[x][y] = [0, 0, 255]
    #cv2.imshow('image', I)
  cv2.imwrite('erimitage-moravec.jpg', I)
  cv2.waitKey(15000)

def moravec_window(a, x, y, u, v):
  a1 = (int(a[x + u - 1, y + v + 1])- int(a[x - 1, y + 1]))**2
  a2 = (int(a[x + u + 0, y + v + 1])- int(a[x + 0, y + 1]))**2
  a3 = (int(a[x + u + 1, y + v + 1])- int(a[x + 1, y + 1]))**2
  a4 = (int(a[x + u - 1, y + v + 0])- int(a[x - 1, y + 0]))**2
  a5 = (int(a[x + u + 0, y + v + 0])- int(a[x + 0, y + 0]))**2
  a6 = (int(a[x + u + 1, y + v + 0])- int(a[x + 1, y + 0]))**2
  a7 = (int(a[x + u - 1, y + v - 1])- int(a[x - 1, y - 1]))**2
  a8 = (int(a[x + u + 0, y + v - 1])- int(a[x + 0, y - 1]))**2
  a9 = (int(a[x + u + 1, y + v - 1])- int(a[x + 1, y - 1]))**2
  res = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8+ a9
  return(res)

moravec('erimitage.jpg')
# testsssss if workz :D