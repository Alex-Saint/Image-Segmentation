import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('street.jpg')

# Turn 2D matrix image into 1D matrix
img2 = img.reshape((-1, 3))

# Convert pixel values to 32 bit floating point
img2 = np.float32(img2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# How many trys to do
attempts = 10
errors = []
for k in range(1, 10):
	# Run k-means on the image
	ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
	# Convert centroid values to unsigned ints
	center = np.uint8(center)

	# Get color labels for each pixel
	res = center[label.flatten()]

	# Shape labels from k-means to be the same as the original image
	res2 = res.reshape((img.shape))

	# Output the final segmented image
	label = 'segmented' + str(k) + '.jpg'
	cv2.imwrite(label, res2)

	# Keep track of wcss for elbow point graph
	errors.append([k, ret])

# Plot elbow point graph
errors = np.array(errors)
plt.title("Elbow Point Graph")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.plot(errors[:,:1], errors[:,1:])
plt.show()