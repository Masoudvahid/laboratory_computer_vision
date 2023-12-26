import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_frames(folder_path):
    frames = []

    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder path '{folder_path}' does not exist.")
        return frames

    # Iterate through files in the folder
    for filename in sorted(os.listdir(folder_path)):
        filepath = os.path.join(folder_path, filename)

        # Check if the file is an image (you can customize this check based on your file types)
        if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            # Read the image using OpenCV
            frame = cv2.imread(filepath)

            # Append the frame to the array
            frames.append(frame)

    return frames

# Given GPS coordinates
gps_coordinates = np.array([
    [+3379.869120, -02235.031680],  # track 2 start
    [+3379.835260, -02235.040000],  # track 2 end
    [+3379.883830, -02235.025080],  # track 3 start
    [+3379.840820, -02235.039400],  # track 3 end

])

# Given video coordinates
video_coordinates = np.array([
    [74, 161],
    [171, 800],
    [229, 146],
    [701, 741],
])

src_points = gps_coordinates
dst_points = video_coordinates

# Assuming src_points and dst_points are numpy arrays of corresponding points
H, _ = cv2.findHomography(src_points, dst_points)

# Warp the video frame
width = 900
height = 900

# Map GPS coordinates to the video frame
video_file = '/home/user/Documents/accurate_speed_calculation/gps/data4/pass2'
frames = load_frames(video_file)

prediction_gps_path = '/home/user/Documents/accurate_speed_calculation/gps/data4/pass2.txt'
gps_xs = []
gps_ys = []
with open(prediction_gps_path, 'r') as input_file:
    for line in input_file:
        data = line.split()
        lat, long = float(data[1]), float(data[2])
        gps_xs.append(lat)
        gps_ys.append(long)

fig, ax = plt.subplots()
x = []
y = []

# Create interpolation functions
interpolate_x = interp1d(gps_coordinates[:, 0], video_coordinates[:, 0], kind='linear', fill_value='extrapolate')
interpolate_y = interp1d(gps_coordinates[:, 1], video_coordinates[:, 1], kind='linear', fill_value='extrapolate')

for gps_x, gps_y, frame in zip(gps_xs, gps_ys, frames):
    frame = cv2.resize(frame, (width, height))
    warped_frame = cv2.warpPerspective(frame, H, (width, height))

    x.append(gps_x)
    y.append(gps_y)

    # Map GPS coordinates to video coordinates using interpolation
    mapped_point = (int(interpolate_x(gps_x)), int(interpolate_y(gps_y)))

    cv2.circle(frame, video_coordinates[0], radius=5, color=(255, 255, 0), thickness=-1)
    cv2.circle(frame, video_coordinates[1], radius=5, color=(255, 255, 0), thickness=-1)
    cv2.circle(frame, video_coordinates[2], radius=5, color=(255, 255, 0), thickness=-1)
    cv2.circle(frame, video_coordinates[3], radius=5, color=(255, 255, 0), thickness=-1)

    cv2.circle(frame, mapped_point, radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow("Frame with Mapped Point", frame)
    cv2.waitKey(250)

    # Plot the mapped point on the graph
    ax.scatter(mapped_point[0], mapped_point[1], c='r')

# Close all windows when the loop is finished
cv2.destroyAllWindows()

fig_gps, ax_gps = plt.subplots()
ax_gps.scatter(x, y, c='r')

plt.show()
