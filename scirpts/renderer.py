import cv2
depth = cv2.imread("/home/ipa-mah/1_projects/object_rendering/build/00000.png",2)
cols,rows = depth.shape
for i in range(rows):
      for j in range(cols):
         k = depth[i,j]
         if(k!=65535):
            print k