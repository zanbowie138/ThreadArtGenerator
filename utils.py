import numpy as np
import cv2 as cv

def calculate_pins(radius: int, num_pins: int):
    # Create empty array of 2d points
    pins = np.empty((num_pins,2), dtype=object)

    # Fill array with pins
    for i in range(num_pins):
        angle = (2*np.pi/num_pins)*i
        x = (int)((radius-3)/2 * np.cos(angle) + radius/2)
        y = (int)((radius-3)/2 * np.sin(angle) + radius/2)
        pins[i] = np.array((x,y))

    return pins

def prepare_image(import_filepath: str, max_resolution: tuple, invert_image: bool = False):
    # Import image
    img = cv.imread(import_filepath, cv.IMREAD_GRAYSCALE)

    # Invert all bits (if background is black)
    if invert_image: img = ~img

    # the smaller side of the image fed to the program will be the size of the cropped image
    crop_size = min(img.shape)

    # crop the image to a square aspect ratio
    cropped_img = img[(int)(img.shape[0]/2 - crop_size/2):(int)(img.shape[0]/2 + crop_size/2), (int)(img.shape[1]/2 - crop_size/2):(int)(img.shape[1]/2 + crop_size/2)]

    # Lower the resolution
    radius = min(crop_size, max_resolution)
    cropped_img = cv.resize(cropped_img, dsize=(radius, radius), interpolation=cv.INTER_CUBIC)
    
    return cropped_img, radius

def image_convolution(matrix, kernel):
    # assuming kernel is symmetric and odd
    k_size = len(kernel)
    m_height, m_width = matrix.shape
    padded = np.pad(matrix, (k_size-1, k_size-1))
    
    # iterates through matrix, applies kernel, and sums
    output_vals = []
    for i in range(m_height):
        for j in range(m_width):
            output_vals.append(np.sum(padded[i:k_size+i, j:k_size+j]*kernel))

    new_img = np.array(output_vals).reshape((m_height, m_width))

    output = np.clip(new_img, 0, 255).astype(np.uint8)

    # Goofy set border to 0
    # probably better way to do this but whatever
    output[:,(0,1)] = 0
    output[(0,1),:] = 0

    return output

def sobel_edge_detection(img):
    # Blur the image for better edge detection
    img_blur = cv.GaussianBlur(img, (3,3), 0)
    
    # Sobel Edge Detection
    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    channel_image = sobelxy

    # Normalize the image
    # https://stackoverflow.com/questions/74861003/rescaling-image-to-get-values-between-0-and-255
    if channel_image.sum()!=0:
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
            
    channel_image = np.clip(channel_image, 0, 255).astype("uint8")
    return channel_image