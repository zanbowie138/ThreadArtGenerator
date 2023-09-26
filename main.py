import sys
import heapq
import cv2 as cv
import numpy as np

# CONSTANTS/CONFIG
IMPORT_FILEPATH = "res/mona_lisa.jpg"
PINNED_FILEPATH = "output/pinned.jpg"
OUTPUT_FILEPATH = "output/output.jpg"
PINS_OUTPUT_FILEPATH = "output/string_path.txt"

NUM_PINS = 100
NUM_LINES = 2000 
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
priority_queues = []
for f, pin_from in enumerate(pins):
    priority_queue = []
    for t, pin_to in enumerate(pins):
        # If calculating a line with itself, continue
        if f == t: continue

        # Create image with line
        line_image = cv.line(np.copy(cropped_img), pin_from, pin_to, 0, 1)

        # Line length
        line_length = np.linalg.norm(np.array(pin_from) - np.array(pin_to))

        # Cost is the difference between the images averaged with line length
        cost = np.sum(cv.absdiff(line_image, cropped_img))/line_length

        # Push cost and destination pin index
        heapq.heappush(priority_queue, (cost, t)) 

    priority_queues.append(priority_queue)
    del priority_queue

    # Fancy status printing
    sys.stdout.write('\r')
    sys.stdout.write(f'Calculating priority queues... {(int)(f/NUM_PINS*100)}%')
    sys.stdout.flush()

# New line out of status
sys.stdout.write("\n")

final_image = np.full((radius, radius), 255, dtype=np.uint8)
start_index = 0

# Draw pins
for p in pins:
    pinned_image = cv.circle(cropped_img, (p[0]-1,p[1]-1), 3, 255, -1)

pin_history = []
for i in range(NUM_LINES):
    this_pq = priority_queues[start_index]

    dest_index = heapq.heappop(this_pq)[1]

    # If destination has their smallest cost line as a duplicate, remove it
    dest_pq = priority_queues[dest_index]
    if (dest_pq[0][1] == start_index and len(dest_pq) > 0):
        heapq.heappop(dest_pq)
    

    # uniqueness = np.sum(cv.absdiff(final_image, cv.line(np.copy(final_image), pins[start_index], pins[dest_index], 255, 1)))
    # if uniqueness < 1000 and len(this_pins_pq) > 0:
    #     continue

    # draw final line on image
    final_image = cv.line(final_image, pins[start_index], pins[dest_index], 0, 1)

    # Add 
    pin_history.append(start_index)

    # set starting index to destination for next loop
    start_index = dest_index

    sys.stdout.write('\r')
    sys.stdout.write(f'Printing image... {(int)(i/NUM_LINES*100)}%')
    sys.stdout.flush()

# Add last pin
pin_history.append(start_index)

print("\nFinished!")

if PREVIEW_IMAGE:
    cv.imshow("pinned image", pinned_image)
    cv.imshow("final image", final_image)

# Write image to output files
if PINNED_FILEPATH != "": cv.imwrite(PINNED_FILEPATH, pinned_image)
if OUTPUT_FILEPATH != "": cv.imwrite(OUTPUT_FILEPATH, final_image)

if PINS_OUTPUT_FILEPATH != "":
    f = open(PINS_OUTPUT_FILEPATH, "w")
    f.write("\n".join([str(p) for p in pin_history]))
    f.close()

# wait until window is closed to stop GUI
cv.waitKey(0)

# clear memory and destroy GUI
cv.destroyAllWindows()