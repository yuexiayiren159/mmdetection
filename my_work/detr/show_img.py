import cv2
import matplotlib.pyplot as plt

img_path = r"../../data/coco4/val2017/000000074058.jpg"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()