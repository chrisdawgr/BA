import harris
import surfdescriptor as sdes
import surfdetector as sdet
import helperfunctions as h
import cv2
import numpy as np

def surf_tester(pic1, pic2):
  p1 = sdet.findSurfPoints(pic1)
  p2 = sdet.findSurfPoints(pic2)
  np.save("markstor1", p1)
  np.save("markstor1", p2)

  p1 = np.load("markstor1.npy")
  p2 = np.load("markstor2.npy")

  desc1po1 = sdes.surf_descriptor(pic1, p1)
  desc2po2 = sdes.surf_descriptor(pic2, p2)

  np.save("markstor1-desc", desc1po1)
  np.save("markstor2-desc", desc2po2)
  # oneNN (des1, des2, p1, p2)

  desc1po1 = np.load("markstor1-desc.npy")
  desc2po2 = np.load("markstor2-desc.npy")
  #(sp1, sp2, d1, d2) = h.oneNN(desc1po1[:,1], desc2po2[:,1], desc1po1[:,0], desc2po2[:,0])

  (sp1, sp2, d1, d2) = h.oneNN_wdist(desc1po1[:,1], desc2po2[:,1], desc1po1[:,0], desc2po2[:,0])
  #(sp1, sp2) = h.advanced_oneNN(desc1po1[:,1], desc2po2[:,1], desc1po1[:,0], desc2po2[:,0])
  #print(sp1[0:10])
  #print("\n")
  #print(sp2[0:10])

  h.drawMatches(pic1, sp1[0:10], pic2, sp2[0:10], [])

  """
  #      test for drawMatches   
  desc1po1 = np.load("room10-desc.npy")
  desc2po2 = np.load("room11-desc.npy")

  desc1po1 = desc1po1[np.argsort(desc1po1[:,0])]
  desc2po2 = desc2po2[np.argsort(desc2po2[:,0])]

  desc1po1 = [desc1po1[523,0]]
  desc2po2 = [desc2po2[327,0]]

  desc1po1.append([0,0,0])
  desc2po2.append([0,0,0])
  h.drawMatches(pic1, desc1po1, pic2, desc2po2, [])
  """


surf_tester("markstor1.jpg", "markstor2.jpg")
