
import cv2

image = cv2.imread("mnist.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image.shape)
# cv2.imshow("", image)
# cv2.waitKey()

count = 0
imnum = 0

for i in range(0, 1000, 20):
    for j in range(0, 2000, 20):
        num = image[i:i+20, j:j+20]
        filepath = "dataset/num" + str(imnum) + "/imnum" + str(imnum) + "_" + str(count) + ".png"
        count += 1
        cv2.imwrite(filepath, num)

    if count == 500:
        imnum += 1
        count = 0
