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
EDGE_OUTPUT_FILEPATH = "output/edges.jpg"

NUM_PINS = 100
NUM_LINES = 1000

MAX_RESOLUTION = 700

INVERT_IMAGE = False
PREVIEW_IMAGE = False
USE_LINE_LENGTH = True
USE_LINE_EFFECTIVENESS = False


# TO RUN ON CODESPACES:
# sudo su
# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# CTRL-SHIFT-P -> Python: Disable Linting (if not already disabled)

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
    sobelx = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
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

def main():
    # Start time
    start_time = time.perf_counter()

    # Get initial image
    base_img, radius = prepare_image(IMPORT_FILEPATH, MAX_RESOLUTION, INVERT_IMAGE)

    # Create edge detection kernel
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Convolute kernel
    # edge_img = ~image_convolution(base_img, kernel)

    edge_img = sobel_edge_detection(base_img)

    # Output edge image
    if EDGE_OUTPUT_FILEPATH != "": cv.imwrite(EDGE_OUTPUT_FILEPATH, edge_img)

    # Calculate positions of pins around the image
    pins = calculate_pins(radius, NUM_PINS)

    # Initialize 2d array of line costs
    ground_truth_costs = np.empty((NUM_PINS, NUM_PINS))

    # Calculate lines' cost based on image
    for f_idx, pos_f in enumerate(pins):
        for t_idx, pos_t in enumerate(pins):
            # If calculating a line with itself, continue
            if f_idx == t_idx: continue

            # Create proposed new image with line
            proposed_img = cv.line(np.copy(base_img), pos_f, pos_t, 0, 1)
            proposed_edge_img = cv.line(np.copy(edge_img), pos_f, pos_t, 0, 1)

            actual_cost = np.sum(cv.absdiff(proposed_img, base_img)).item()
            actual_edge_cost = np.sum(cv.absdiff(proposed_edge_img, edge_img)).item()
            # print(f'Actual cost: {actual_cost}, Actual edge cost: {actual_edge_cost}')

            # Cost is the difference between the images
            cost = actual_cost - actual_edge_cost * 10
            cost = -actual_edge_cost

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

    # Pin history of every pin (to prevent duplicate lines)
    per_pin_history = [set() for x in range(NUM_PINS)]

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

            # Add line to proposed image
            proposed_line_img = cv.line(np.copy(final_image), from_pos, dest_pos, 0, 1)

            # See how much line actually changes the picture
            line_effectiveness = np.sum(cv.absdiff(proposed_line_img, final_image)) 

            # Calculate line length
            line_length = np.linalg.norm(np.array(from_pos) - np.array(dest_pos))

            # Calculate actual cost
            scaled_cost = (ground_truth_costs[from_idx, dest_idx])

            if USE_LINE_EFFECTIVENESS: scaled_cost -= line_effectiveness.item()

            # If true, higher length lines will be more prevalent at the cost of a possibly worse image
            if USE_LINE_LENGTH: scaled_cost /= line_length.item()

            # Push cost and destination pin index
            heapq.heappush(priority_queue, (scaled_cost, dest_idx))

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