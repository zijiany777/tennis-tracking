import cv2
import numpy as np

def gaussian_sobel_magnitude(image, sigma):
    # Apply Gaussian filter
    blurred = cv2.GaussianBlur(image, (0,0), sigmaX=sigma, sigmaY=sigma)

    # Calculate the x and y derivatives using Sobel filters
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of the gradient
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    return magnitude


# def gaussian_derivative(image, sigma, order, axis='x'):
#     # Apply Gaussian filter
#     blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
#
#     # Calculate the derivative along the specified axis
#     if axis == 'x':
#         derivative = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
#     elif axis == 'y':
#         derivative = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
#     else:
#         raise ValueError("Axis must be 'x' or 'y'.")
#
#     # Apply the derivative order
#     result = derivative if order == 1 else derivative if order == 2 else None
#
#     return result


#Example usage
image = cv2.imread('greencourt96.png', cv2.IMREAD_GRAYSCALE)
sigma_value = 100
result = gaussian_sobel_magnitude(image, sigma=sigma_value)

# image = cv2.imread('bluecourt86.png', cv2.IMREAD_GRAYSCALE)
# sigma_value = 10
# order_value = 1
# axis_value = 'x'
# result = gaussian_derivative(image, sigma=sigma_value, order=order_value, axis=axis_value)

# Display the original and result images
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Sobel Magnitude', result)
cv2.waitKey(0)
cv2.destroyAllWindows()