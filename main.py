import numpy as np

import matplotlib.pyplot as plt
from scipy import misc # contains an image of a racoon!
from PIL import Image # for reading image files

print("""
#################################
Introduction to arrays with numpy
#################################""")

#1D array
my_array = np.array([1.1, 9.2, 8.1, 4.7])
print(f"Data array is: {my_array}")
print(f"Shape is: {my_array.shape}")
print(f"Third element is {my_array[2]}")
print(f"Dimension of the array is: {my_array.ndim}")
print("")

#2D array (matrix)
array_2d = np.array([[1, 2, 3, 9], [5, 6, 7, 8]])

print(f'array_2d has {array_2d.ndim} dimensions')
print(f'Its shape is {array_2d.shape}')
print(f'It has {array_2d.shape[0]} rows and {array_2d.shape[1]} columns')
print(array_2d)
#Access a value
print(f"Getting value in index [1,2]: {array_2d[1,2]}")
#Access a row
print("Getting the first row, [0, :]: {array_2d[0, :]}")
print("")

#n-dim arrays (tensors)
mystery_array = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                         [[7, 86, 6, 98],
                          [5, 1, 0, 4]],

                          [[5, 36, 32, 48],
                           [97, 0, 27, 18]]])
print(f'We have {mystery_array.ndim} dimensions')
print(f'The shape is {mystery_array.shape}')
print(f"This is the last element: {mystery_array[2,1,3]}")
print(f"This is the last row: {mystery_array[2,1,:]}")
#Try to retrieve a (3,2) matrix with the values [[0, 4], [7, 5], [5, 97]]
print("Getting the requested matrix (first element of 3rd index):")
print (mystery_array[:,:,0])
print(f"Original array was: {mystery_array}")

print("""
#################################
End of introduction
#################################""")
print("")

print("""
#################################
Introduction to arrays with numpy
#################################""")
print("Create a vector a with values ranging from 10 to 29")
a = np.arange(10,30)
print(a)

print("Create an array containing only the last 3 values of a")
print(a[-3:])

print("Create a subset with only the 4th, 5th, and 6th values")
print(a[3:6])

print("Create a subset of a containing all the values except for the first 12 (i.e., [22, 23, 24, 25, 26, 27, 28, 29])")
print(a[12:])

print("Create a subset that only contains the even numbers (i.e, every second number)")
print(a[1::2])

print("Reverse the order of the values in a, so that the first element comes last:")
print(a[::-1])
#print(np.flip(a))

print("Print out all the indices of the non-zero elements in this array: [6,0,9,0,0,5,0]")
b = np.array([6,0,9,0,0,5,0])
print(np.nonzero(b))

print("")
print("Use NumPy to generate a 3x3x3 array with random numbers")
print(np.random.random((3, 3, 3)))

print("")
print("Use .linspace() to create a vector x of size 9 with values spaced out evenly between 0 to 100 (both included).")
x = np.linspace(0, 100, 9)
print(x)

print("")
print("Use .linspace() to create another vector y of size 9 with values between -3 to 3 (both included). Then plot x and y on a line chart using Matplotlib.")
y = np.linspace(-3, 3, 9)
plt.plot(x, y)
plt.show()

print("")
print("""Use NumPy to generate an array called noise with shape 128x128x3 that
has random values. Then use Matplotlib's .imshow() to display the array as an image.""")
print("The random values will be interpreted as the RGB colours for each pixel.")
noise = np.random.random((128, 128, 3))
plt.imshow(noise)
plt.show()

print("")
print("""
###############################################
Broadcasting, Scalars and Matrix Multiplication
###############################################""")
a1 = np.array([[1, 3],
               [0, 1],
               [6, 2],
               [9, 7]])

b1 = np.array([[4, 1, 3],
               [5, 8, 5]])
print("Matrix multiplication")
print(f"a1 = {a1}")
print(f"b1 = {b1}")
print("a1 · b1 = ")
print(a1 @ b1)
#print(np.matmul(a1, b1))
print("")

print("""
##############################################
Image manipulation as n-arrays
##############################################""")
img = misc.face()
plt.imshow(img)
plt.show()
print("""What is the data type of img?
Also, what is the shape of img and how many dimensions does it have?
What is the resolution of the image?
      """)
print(type(img))
print(img.shape)
print(img.ndim)
print("Now can you try and convert the image to black and white")
sRGB_array = img / 255
grey_vals = np.array([0.2126, 0.7152, 0.0722])
img_gray = sRGB_array @ grey_vals
#img_gray = np.matmul(sRGB_array, grey_vals)
plt.imshow(img_gray, cmap='gray')
plt.show()
print("""
      Can you manipulate the images by doing some operations on the underlying ndarrays? See if you can change the values in the ndarray so that:
      """)
print("Flip the grayscale image upside down:")
plt.imshow(np.flip(img_gray), cmap='gray')
plt.show()
print("Rotate the colour image:")
plt.imshow(np.rot90(img))
plt.show()
print("""
Invert (i.e., solarize) the colour image.To do this you need to convert
all the pixels to their "opposite" value, so black (0) becomes white (255)""")
plt.imshow(255 - img)
plt.show()
print("""
      Load your own image and manipulate it!
      """)
my_img = Image.open("yummy_macarons.jpg")
img_array = np.array(my_img)
plt.imshow(img_array)
plt.show()
plt.imshow(255 - img_array)
plt.show()
