from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


car_image = imread('test-img/car6.jpg', as_gray=True)
print(car_image.shape)

gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
ax2.imshow(binary_car_image, cmap="gray")
plt.show()
print('hello')
print('hello')
print('hello')