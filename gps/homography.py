import cv2
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt


# cv2.imshow('a', cv2.resize(cv2.imread('/home/user/Documents/accurate_speed_calculation/gps/data4/calibration/pass1/1690888162050.jpg'), (960,960)))
# cv2.waitKey(0)
# cv2.imshow('a', cv2.resize(cv2.imread('/home/user/Documents/accurate_speed_calculation/gps/data4/calibration/pass1/1690888165600.jpg'), (960,960)))
# cv2.waitKey(0)
# cv2.imshow('a', cv2.resize(cv2.imread('/home/user/Documents/accurate_speed_calculation/gps/data4/calibration/pass3/1690888856550.jpg'), (960,960)))
# cv2.waitKey(0)
# cv2.imshow('a', cv2.resize(cv2.imread('/home/user/Documents/accurate_speed_calculation/gps/data4/calibration/pass3/1690888860150.jpg'), (960,960)))
# cv2.waitKey(0)

# cv2.imshow('a', cv2.resize(cv2.imread('/home/user/Documents/accurate_speed_calculation/gps/data1_back/calibration/lane2/1681456230350.jpg'), (960,960)))
# cv2.waitKey(0)
# cv2.imshow('a', cv2.resize(cv2.imread('/home/user/Documents/accurate_speed_calculation/gps/data1_back/calibration/lane2/1681456233500.jpg'), (960,960)))
# cv2.waitKey(0)
# cv2.imshow('a', cv2.resize(cv2.imread('/home/user/Documents/accurate_speed_calculation/gps/data1_back/calibration/lane3/1681456503200.jpg'), (960,960)))
# cv2.waitKey(0)
# cv2.imshow('a', cv2.resize(cv2.imread('/home/user/Documents/accurate_speed_calculation/gps/data1_back/calibration/lane3/1681456505500.jpg'), (960,960)))
# cv2.waitKey(0)

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
    [+3379.869120, -02235.031680],  # track 1 start
    [+3379.835260, -02235.040000],  # track 1 end
    [+3349.556808, -02214.208620],  # track 3 start
    [+3349.568160, -02214.138150]  # track 3 end
])

video_coordinates = np.array([
    [621, 9],
    [120, 412],
    [872, 10],
    [554, 673],
])

src_points = gps_coordinates
dst_points = video_coordinates

# Assuming src_points and dst_points are numpy arrays of corresponding points
H, _ = cv2.findHomography(src_points, dst_points)

# Warp the video frame
width = 960
height = 960

# Map GPS coordinates to the video frame
video_file = '/home/user/Documents/accurate_speed_calculation/gps/data1_back/lane1'

frames = load_frames(video_file)

prediction_gps_path = '/home/user/Documents/accurate_speed_calculation/gps/data1_back/lane1.txt'
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

for gps_x, gps_y, frame in zip(gps_xs, gps_ys, frames):
    frame = cv2.resize(frame, (width, height))
    warped_frame = cv2.warpPerspective(frame, H, (width, height))

    x.append(gps_x)
    y.append(gps_y)

    mapped_point = np.dot(H, np.array([gps_x, gps_y, 1]))
    mapped_point = (mapped_point / mapped_point[2])[:2]
    # mapped_point_int = tuple(mapped_point.astype(int))
    mapped_point_int = tuple(np.round(mapped_point).astype(int))

    cv2.circle(frame, video_coordinates[0], radius=5, color=(0, 0, 255), thickness=-1)  # First - Red
    cv2.circle(frame, video_coordinates[1], radius=5, color=(0, 255, 0), thickness=-1)  # Second - Green
    cv2.circle(frame, video_coordinates[2], radius=5, color=(255, 0, 0), thickness=-1)  # Third - Blue
    cv2.circle(frame, video_coordinates[3], radius=5, color=(0, 0, 0), thickness=-1)  # Fourth - Black

    cv2.circle(frame, (mapped_point_int[0], mapped_point_int[1]), radius=5, color=(255, 255, 255), thickness=-1)

    cv2.imshow("Frame with Mapped Point", frame)
    cv2.waitKey(250)

    # Plot the mapped point on the graph
    ax.scatter(mapped_point_int[0], mapped_point_int[1], c='r')

    # time.sleep(0.65)ax_gps
cv2.destroyAllWindows()
fig_gps, ax_gps = plt.subplots()
ax_gps.scatter(x, y, c='r')

plt.show()
