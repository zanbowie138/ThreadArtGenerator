import sys
import heapq
import cv2 as cv
import numpy as np
import time

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
USE_LINE_LENGTH = False


# TO RUN ON CODESPACES:
# sudo su
# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# CTRL-SHIFT-P -> Python: Disable Linting (if not already disabled)

# Start time
start_time = time.perf_counter()

# Import image
img = cv.imread(IMPORT_FILEPATH, cv.IMREAD_GRAYSCALE)

# Invert all bits (if background is black)
if INVERT_IMAGE: img = ~img

# the smaller side of the image fed to the program will be the size of the cropped image
crop_size = min(img.shape)

# crop the image to a square aspect ratio
cropped_img = img[(int)(img.shape[0]/2 - crop_size/2):(int)(img.shape[0]/2 + crop_size/2), (int)(img.shape[1]/2 - crop_size/2):(int)(img.shape[1]/2 + crop_size/2)]

# Lower the resolution
radius = min(crop_size, MAX_RESOLUTION)
cropped_img = cv.resize(cropped_img, dsize=(radius, radius), interpolation=cv.INTER_CUBIC)

# create a circle of pins around the image
pins = []
for i in range(NUM_PINS):
    angle = (2*np.pi/NUM_PINS)*i
    x = (int)((radius-3)/2 * np.cos(angle) + radius/2)
    y = (int)((radius-3)/2 * np.sin(angle) + radius/2)
    pins.append((x,y))

# TODO: add convolutional edge detection

# Initialize 2d array of line costs
ground_truth_costs = np.empty((NUM_PINS, NUM_PINS))

# Calculate lines' cost based on image
for f_idx, pos_f in enumerate(pins):
    for t_idx, pos_t in enumerate(pins):
        # If calculating a line with itself, continue
        if f_idx == t_idx: continue

        # Create proposed new image with line
        proposed_img = cv.line(np.copy(cropped_img), pos_f, pos_t, 0, 1)

        # Cost is the difference between the images
        cost = np.sum(cv.absdiff(proposed_img, cropped_img)).item()

        # Add cost to array
        ground_truth_costs[f_idx, t_idx] = cost
    
    # Fancy status printing
    sys.stdout.write('\r')
    sys.stdout.write(f'Calculating ground truth costs... {(int)(f_idx/NUM_PINS*100)}%')
    sys.stdout.flush()

# Move to next line
sys.stdout.write("\n")

# Create blank final image
final_image = np.full((radius, radius), 255, dtype=np.uint8)

from_idx = 0
pin_history = [set() for x in range(NUM_PINS)]

# Calculate final lines
for line_idx in range(NUM_LINES):
    from_pos = pins[from_idx]

    # Rank each possible line by cost
    priority_queue = []
    for dest_idx, dest_pos in enumerate(pins):
        # If calculating a line with itself, continue
        if from_idx == dest_idx: continue

        # Add line to proposed image
        proposed_line_img = cv.line(np.copy(final_image), from_pos, dest_pos, 0, 1)

        # See how much line actually changes the picture
        line_effectiveness = np.sum(cv.absdiff(proposed_line_img, final_image)) 

        # Line length
        line_length = np.linalg.norm(np.array(from_pos) - np.array(dest_pos))

        # Calculate actual cost
        scaled_cost = (ground_truth_costs[from_idx, dest_idx] - line_effectiveness.item())

        # If true, higher length lines will be more prevalent at the cost of a possibly worse image
        if USE_LINE_LENGTH: scaled_cost /= line_length.item()

        # Push cost and destination pin index
        heapq.heappush(priority_queue, (scaled_cost, dest_idx))


    # Only create new lines
    heap_idx = 1
    dest_index = priority_queue[0][1]
    while (True):
        # If there are no new lines, raise exception
        if heap_idx == len(priority_queue) - 1:
            raise Exception("Pin duplicate!")
        
        # If the line has already been drawn, try next best
        elif dest_index in pin_history[from_idx]:
            dest_index = priority_queue[heap_idx][1]
            heap_idx+=1

        # Otherwise, the line is new 
        else:
            break

    # draw final line on image
    final_image = cv.line(final_image, pins[from_idx], pins[dest_index], 0, 1)

    # add destination index to history
    pin_history[from_idx].add(dest_index)
    pin_history[dest_index].add(from_idx)

    # set starting index to destination for next loop
    from_idx = dest_index

    # Fancy status printing
    sys.stdout.write('\r')
    sys.stdout.write(f'Printing lines... {(int)(line_idx/NUM_LINES*100)}%')
    sys.stdout.flush()


# Draw pins
for p in pins:
    pinned_image = cv.circle(cropped_img, (p[0]-1,p[1]-1), 3, 255, -1)

print(f'\nFinished! \nProgram time: {time.perf_counter() - start_time:0.4f} seconds')

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