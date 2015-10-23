import harris
import surfdescriptor as sd
import helperfunctions as h
import cv2
import numpy as np


def surf_tester(pic1, pic2):
  p1 = harris.harris(pic1, 0.04, 390000000, 1)
  p2 = harris.harris(pic2, 0.04, 380000000, 1)

  pp1 = []
  pp2 = []

  #Appending a scale to the point, eg: [y,x,s]
  for i in p1:
    pp1.append([i[0], i[1], 1.2])
    
  for i in p2:
    pp2.append([i[0], i[1], 1.2])


  (desc1, po1) = sd.surf_descriptor(pic1, pp1)
  (desc2, po2) = sd.surf_descriptor(pic2, pp2)

  (sp1, sp2) = h.oneNN(desc1, desc2, po1, po2)


  for i in pp1:
    po1.append([i[0], i[1]])

  for i in pp2:
    po2.append([i[0], i[1]])

  h.drawMatches(pic1, sp1, pic2, sp2, [])

#surf_tester("erimitages2.jpg", "erimitages1.jpg")
