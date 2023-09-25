import heapq
import cv2 as cv
import numpy as np
import sys

img = cv.imread("mona_lisa.jpg", cv.IMREAD_GRAYSCALE)

# the smaller side of the image fed to the program will be the radius of the circle
radius = min(img.shape)

# crop the image to a square with side length equal to the radius.
cropped_img = img[(int)(img.shape[0]/2 - radius/2):(int)(img.shape[0]/2 + radius/2), (int)(img.shape[1]/2 - radius/2):(int)(img.shape[1]/2 + radius/2)]

# the amount of 'nails' to be used in the image
num_pins = 100
pins = []

# create a circle of pins around the image
for i in range(num_pins):
    angle = (2*np.pi/num_pins)*i
    x = (int)((radius-3)/2 * np.cos(angle) + radius/2)
    y = (int)((radius-3)/2 * np.sin(angle) + radius/2)
    pins.append((x,y))


# calculate priority queues for each pin

# NOTE: will use a linked list in the future when I consider the line against previous lines, not just the previous image
priority_queues = []
for f, pin_from in enumerate(pins):
    priority_queue = []
    for t, pin_to in enumerate(pins):
        # If calculating a line with itself, continue
        if f == t:
            continue

        line_image = cv.line(np.copy(cropped_img), pin_from, pin_to, 0, 1)

        # Line length
        line_length = np.linalg.norm(np.array(pin_from) - np.array(pin_to))

        # Cost is the difference between the images averaged with line length
        COMPUTED_COST = np.sum(cv.absdiff(line_image, cropped_img)) /line_length

        # Push cost and destination pin index
        heapq.heappush(priority_queue, (COMPUTED_COST, t)) 

        del line_image

    priority_queues.append(priority_queue)
    del priority_queue

    sys.stdout.write('\r')
    sys.stdout.write(f'Calculating priority queues... {(int)(f/num_pins*100)}%')
    sys.stdout.flush()

sys.stdout.write("\n")

final_image = np.full((radius, radius), 255, dtype=np.uint8)
num_lines = 2000
start_index = 0

# Draw pins
for p in pins:
    pinned_image = cv.circle(cropped_img, (p[0]-1,p[1]-1), 3, 255, -1)

for i in range(num_lines):
    this_pins_pq = priority_queues[start_index]
    # TODO: don't just pop. Pop and see if it's different enough from what we have already drawn
    dest_index = heapq.heappop(this_pins_pq)[1]

    # If destination has their smallest cost line as a duplicate, remove it
    dest_pq = priority_queues[dest_index]
    if (dest_pq[0][1] == start_index and len(dest_pq) > 0):
        heapq.heappop(dest_pq)
    

    # uniqueness = np.sum(cv.absdiff(final_image, cv.line(np.copy(final_image), pins[start_index], pins[dest_index], 255, 1)))
    # if uniqueness < 1000 and len(this_pins_pq) > 0:
    #     continue

    final_image = cv.line(final_image, pins[start_index], pins[dest_index], 0, 1)
    start_index = dest_index

    sys.stdout.write('\r')
    sys.stdout.write(f'Printing image... {(int)(i/num_lines*100)}%')
    sys.stdout.flush()

print("\nFinished!")

cv.imshow("pinned image", pinned_image)
cv.imshow("final image", final_image)

# cv.imwrite("pinned.jpg", pinned_image)
# cv.imwrite("output.jpg", final_image)

# wait until window is closed to stop GUI
cv.waitKey(0)

# clear memory and destroy GUI
cv.destroyAllWindows()