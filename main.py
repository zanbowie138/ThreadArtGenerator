import sys
import heapq
import cv2 as cv
import numpy as np

# CONSTANTS/CONFIG
IMPORT_FILEPATH = "res/mona_lisa.jpg"
PINNED_FILEPATH = "output/pinned.jpg"
OUTPUT_FILEPATH = "output/output.jpg"

NUM_PINS = 20
NUM_LINES = 100 
MAX_RESOLUTION = 700

INVERT_IMAGE = False
PREVIEW_IMAGE = False


# TO RUN ON CODESPACES:
# sudo su
# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# CTRL-SHIFT-P -> Python: Disable Linting (if not already disabled)

# Import image
img = cv.imread(IMPORT_FILEPATH, cv.IMREAD_GRAYSCALE)

# Invert all bits (if background is black)
if INVERT_IMAGE:
    img = ~img

# the smaller side of the image fed to the program will be the size of the cropped image
crop_size = min(img.shape)

# crop the image to a square aspect ratio
cropped_img = img[(int)(img.shape[0]/2 - crop_size/2):(int)(img.shape[0]/2 + crop_size/2), (int)(img.shape[1]/2 - crop_size/2):(int)(img.shape[1]/2 + crop_size/2)]

# Lower the resolution
radius = min(crop_size, MAX_RESOLUTION)
cropped_img = cv.resize(cropped_img, dsize=(radius, radius), interpolation=cv.INTER_CUBIC)

# the amount of 'nails' to be used in the image
pins = []

# create a circle of pins around the image
for i in range(NUM_PINS):
    angle = (2*np.pi/NUM_PINS)*i
    x = (int)((radius-3)/2 * np.cos(angle) + radius/2)
    y = (int)((radius-3)/2 * np.sin(angle) + radius/2)
    pins.append((x,y))

# TODO: Next algorithm will be to consider all other lines
# Don't precalculate priority queues for each pin, but rather bruteforce all combinations every pin step

# TODO: add convolutional edge detection
# TODO: Fix crashes/edge cases
# TODO: Actually prevent duplicate lines

# calculate priority queues for each pin
start_index = 0
line_image = np.full(cropped_img.shape,255, dtype=np.uint8)
final_image = np.full((radius, radius), 255, dtype=np.uint8)
pin_history = [[] for x in range(NUM_PINS)]

for line_idx in range(NUM_LINES):
    priority_queue = []
    pin_from = pins[start_index]

    for t, pin_to in enumerate(pins):
        # If calculating a line with itself, continue
        if start_index == t: continue

        # Add line to proposed image
        proposed_line_img = cv.line(np.copy(line_image), pin_from, pin_to, 0, 1)

        # Actual line change
        line_effectiveness = np.sum(cv.absdiff(proposed_line_img, line_image))

        # Line length
        line_length = np.linalg.norm(np.array(pin_from) - np.array(pin_to))

        # Cost is the difference between the images averaged with line length
        benefit = (np.sum(cv.absdiff(line_image, cropped_img))) / (line_length)

        # Push benefit and destination pin index
        heapq.heappush(priority_queue, (benefit, t)) 

    dest_index = heapq.heappop(priority_queue)[1]
    while (dest_index in pin_history[start_index]):
        dest_index = heapq.heappop(priority_queue)[1]

    # draw final line on image
    final_image = cv.line(final_image, pins[start_index], pins[dest_index], 0, 1)

    # draw line on line image
    line_image = cv.line(line_image, pins[start_index], pins[dest_index], 0, 1)

    # add destination index to history
    pin_history[start_index].append(dest_index)
    pin_history[dest_index].append(start_index)

    # set starting index to destination for next loop
    start_index = dest_index

    # Fancy status printing
    sys.stdout.write('\r')
    sys.stdout.write(f'Printing lines... {(int)(line_idx/NUM_LINES*100)}%')
    sys.stdout.flush()


# Draw pins
for p in pins:
    pinned_image = cv.circle(cropped_img, (p[0]-1,p[1]-1), 3, 255, -1)

print("\nFinished!")

if PREVIEW_IMAGE:
    cv.imshow("pinned image", pinned_image)
    cv.imshow("final image", final_image)

# Write image to output files
cv.imwrite(PINNED_FILEPATH, pinned_image)
cv.imwrite(OUTPUT_FILEPATH, final_image)

# wait until window is closed to stop GUI
cv.waitKey(0)

# clear memory and destroy GUI
cv.destroyAllWindows()