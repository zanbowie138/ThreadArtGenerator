import math
import sys
import heapq
import cv2 as cv
import numpy as np
import time

import utils

# CONSTANTS/CONFIG
IMPORT_FILEPATH = "res/abraham_lincoln.webp"

PINNED_FILEPATH = "output/pinned.jpg"
OUTPUT_FILEPATH = "output/output.jpg"
PINS_OUTPUT_FILEPATH = "output/string_path.txt"
EDGE_OUTPUT_FILEPATH = "debug/edges.jpg"

NUM_PINS = 100
NUM_LINES = 2000

MIN_PIN_DISTANCE = 0

LINE_THICKNESS = 2

MAX_RESOLUTION = 700

INVERT_IMAGE = False
PREVIEW_IMAGE = True
USE_LINE_LENGTH = False
USE_LINE_EFFECTIVENESS = False

# TO RUN ON CODESPACES:
# sudo su
# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# CTRL-SHIFT-P -> Python: Disable Linting (if not already disabled)

def main():
    # Start time
    start_time = time.perf_counter()

    # Prepare initial image
    base_img, radius = utils.prepare_image(IMPORT_FILEPATH, MAX_RESOLUTION, INVERT_IMAGE)

    # Create edge detection kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolute kernel
    # edge_img = ~image_convolution(base_img, kernel)

    edge_img = ~utils.sobel_edge_detection(base_img)

    # Debug edge image
    if EDGE_OUTPUT_FILEPATH != "": cv.imwrite(EDGE_OUTPUT_FILEPATH, edge_img)

    # Calculate positions of pins around the image
    pins = utils.calculate_pins(radius, NUM_PINS)

    # Create weighted images
    scaled_edge_img = (~np.copy(edge_img) * 0.0).astype(np.uint8)
    scaled_base_img = (~np.copy(base_img) * 1.0).astype(np.uint8)

    cv.imwrite("debug/scaled_edge_img.png", ~scaled_edge_img)
    cv.imwrite("debug/scaled_base_img.png", ~scaled_base_img)

    # Add weighted images together
    edge_prio_img = cv.add(scaled_base_img,scaled_edge_img)
    cv.imwrite("debug/edge_prio.png", edge_prio_img)

    # Initialize 2d array of line costs
    ground_truth_costs = np.empty((NUM_PINS, NUM_PINS))

    # Initialize 2d array of line lengths
    line_lengths = np.empty((NUM_PINS, NUM_PINS))

    # Calculate lines' cost based on image
    for f_idx, pos_f in enumerate(pins):
        for t_idx, pos_t in enumerate(pins):
            # If calculating a line with itself, continue
            if f_idx == t_idx: continue

            # If pins are too close to each other, continue
            if abs(f_idx - t_idx) < MIN_PIN_DISTANCE: continue

            # Add line to proposed image
            proposed_img = cv.line(np.copy(edge_prio_img), pos_f, pos_t, 0, LINE_THICKNESS)

            # Compare proposed image to actual image
            # The higher the cost, the less good a specific line is
            diff = cv.subtract(proposed_img, edge_prio_img)
            actual_cost = math.sqrt(np.sum(diff**2).item())
            # cv.imwrite("edge.png", cv.absdiff(proposed_img, proposed_edge_img))

            line_length = np.linalg.norm(np.array(pos_f) - np.array(pos_t)).item()
            # print(line_length * 255 - actual_cost)

            # Cost is the difference between the images
            cost = actual_cost

            # print(cost)
            # input("Press Enter to continue...")

            # Add cost to array
            ground_truth_costs[f_idx, t_idx] = -cost

            # Add line length to array
            line_lengths[f_idx, t_idx] = np.linalg.norm(np.array(pos_f) - np.array(pos_t)).item()
        
        # Fancy status printing
        sys.stdout.write('\r')
        sys.stdout.write(f'Calculating ground truth costs... {(int)(f_idx/NUM_PINS*100)}%')
        sys.stdout.flush()

    # Move to next line
    sys.stdout.write("\n")

    # Create blank final image
    final_image = np.full((radius, radius), 255, dtype=np.uint8)

    # Pin history of every pin (to prevent duplicate lines)
    per_pin_history = [set() for _ in range(NUM_PINS)]

    # Overall pin history (for output)
    pin_history = np.empty(NUM_LINES+1)

    # Calculate final lines
    from_idx = 0
    pin_history[0] = from_idx
    for line_idx in range(NUM_LINES):
        from_pos = pins[from_idx]

        # Rank each possible line by cost
        priority_queue = []
        for dest_idx, dest_pos in enumerate(pins):
            # If calculating a line with itself, continue
            if from_idx == dest_idx: continue

            # If pins are too close to each other, continue
            if abs(from_idx - dest_idx) < MIN_PIN_DISTANCE: continue

            # Get cost from original image
            cost = ground_truth_costs[from_idx, dest_idx]

            if USE_LINE_EFFECTIVENESS: 
                # Add line to proposed image
                proposed_line_img = cv.line(np.copy(final_image), from_pos, dest_pos, 0, LINE_THICKNESS)

                # See how much line actually changes the picture
                line_effectiveness = np.sum(cv.absdiff(proposed_line_img, final_image)) 

                # Subtract cost
                cost -= (line_effectiveness.item() * 0.1)

            # If true, higher length lines will be more prevalent at the cost of a possibly worse image
            if USE_LINE_LENGTH: 
                cost /= line_lengths[from_idx, dest_idx]

            # Push cost and destination pin index
            heapq.heappush(priority_queue, (cost, dest_idx))

        # Only create new lines
        heap_idx = 1
        dest_index = priority_queue[0][1]
        while True:
            # If the line has already been drawn, try next best
            if dest_index in per_pin_history[from_idx]:
                dest_index = priority_queue[heap_idx][1]
                heap_idx+=1

            # Otherwise, the line is new 
            else:
                break

        # draw final line on image
        final_image = cv.line(final_image, pins[from_idx], pins[dest_index], 0, 1)

        # add destination index to history
        per_pin_history[from_idx].add(dest_index)
        per_pin_history[dest_index].add(from_idx)

        # Add to overall pin history
        pin_history[line_idx + 1] = dest_index

        # set starting index to destination for next loop iteration
        from_idx = dest_index

        # Fancy status printing
        sys.stdout.write('\r')
        sys.stdout.write(f'Printing lines... {(int)(line_idx/NUM_LINES*100)}%')
        sys.stdout.flush()


    # Draw pins
    for p in pins:
        pinned_image = cv.circle(base_img, (p[0]-1,p[1]-1), 3, 255, -1)

    print(f'\nFinished! \nProgram time: {time.perf_counter() - start_time:0.4f} seconds')

    # Preview image
    if PREVIEW_IMAGE:
        cv.imshow("pinned image", pinned_image)
        cv.imshow("final image", final_image)

    # Write image to output files
    if PINNED_FILEPATH != "": cv.imwrite(PINNED_FILEPATH, pinned_image)
    if OUTPUT_FILEPATH != "": cv.imwrite(OUTPUT_FILEPATH, final_image)

    # Write pin history to txt file
    if PINS_OUTPUT_FILEPATH != "":
        f = open(PINS_OUTPUT_FILEPATH, "w")
        f.write(",".join([str(int(p)) for p in pin_history]))
        f.close()

    # wait until window is closed to stop GUI
    cv.waitKey(0)

    # clear memory and destroy GUI
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()