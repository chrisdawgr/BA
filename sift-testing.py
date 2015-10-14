import SIFT
import orientat
import descriptor

def siftrun(I):
  p1 = SIFT.SIFT(I, 5)
  #print(p1)
  p2 = orientat.sift_orientation(I, p1, 5)
  desc = descriptor.decriptor_representation(I, p2)
  return(desc)

desc1 = siftrun("mandela1.jpg")
desc2 = siftrun("mandela2.jpg")

p = desc1[len(desc1)/2.0]

#for i in desc2:

