import harris
import surfdescriptor as sdes
import surfdetector as sdet
import helperfunctions as h
import cv2
import numpy as np


#
def surf_tester(pic1, pic2):
  p1 = sdet.findSurfPoints(pic1)
  p2 = sdet.findSurfPoints(pic2)

  (desc1, po1) = sdes.surf_descriptor(pic1, p1)
  (desc2, po2) = sdes.surf_descriptor(pic2, p2)

  #(sp1, sp2, d1, d2) = h.oneNN(desc1, desc2, po1, po2)

  (sp1, sp2) = h.advanced_oneNN(desc1, desc2, po1, po2)

  h.drawMatches(pic1, sp1, pic2, sp2, [])

surf_tester("mark-seg1.jpg", "mark-seg2.jpg")
