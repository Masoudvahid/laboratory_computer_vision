import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


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


gps_coordinates = np.array([
    [+03349.559028, -02214.213222],  # track 2 start
    [+03349.569012, -02214.152352],  # track 2 end
    [+03349.557060, -02214.207102],  # track 3 start
    [+03349.566708, -02214.147402]   # track 3 end
])

# Example pixel coordinates in the video frame
video_coordinates = np.array([
    [618, 24],
    [102, 454],
    [870, 27],
    [671, 465]
])

src_points = gps_coordinates
dst_points = video_coordinates

# Assuming src_points and dst_points are numpy arrays of corresponding points
H, _ = cv2.findHomography(src_points, dst_points)

# Warp the video frame
width = 960
height = 960

# Map GPS coordinates to the video frame
gps_file = '/home/user/Documents/accurate_speed_calculation/gps/lane3.csv'
video_file = '/home/user/Downloads/pass3'

gps_data = pd.read_csv(gps_file)
frames = load_frames(video_file)

gps_xs = gps_data.iloc[:, 1].values.reshape(-1, 1)  # Assuming the second column is latitude
gps_ys = gps_data.iloc[:, 2].values.reshape(-1, 1)  # Assuming the third column is longitude

fig, ax = plt.subplots()

x = []
y = []

for gps_x, gps_y, frame in zip(gps_xs, gps_ys, frames):
    frame = cv2.resize(frame, (width, height))
    warped_frame = cv2.warpPerspective(frame, H, (width, height))

    gps_x = gps_x[0]
    x.append(gps_x)
    gps_y = gps_y[0]
    y.append(gps_y)

    mapped_point = np.dot(H, np.array([gps_x, gps_y, 1]))
    mapped_point = (mapped_point / mapped_point[2])[:2]
    mapped_point_int = tuple(np.round(mapped_point).astype(int))

    cv2.circle(frame, video_coordinates[0], radius=5, color=(255, 255, 0), thickness=-1)
    cv2.circle(frame, video_coordinates[1], radius=5, color=(255, 255, 0), thickness=-1)
    cv2.circle(frame, video_coordinates[2], radius=5, color=(255, 255, 0), thickness=-1)
    cv2.circle(frame, video_coordinates[3], radius=5, color=(255, 255, 0), thickness=-1)

    cv2.circle(frame, mapped_point_int, radius=5, color=(0, 255, 0), thickness=-1)

    cv2.imshow("Frame with Mapped Point", frame)
    cv2.waitKey(250)

    # Plot the mapped point on the graph
    ax.scatter(mapped_point_int[0], mapped_point_int[1], c='r')

    # time.sleep(0.65)ax_gps
cv2.destroyAllWindows()
fig_gps, ax_gps = plt.subplots()
ax_gps.scatter(x, y, c='r')

plt.show()
