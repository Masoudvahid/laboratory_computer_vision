import cv2
import numpy as np
import math

points_coordinates = np.array([
    [-9, 19],
    [16, 0],
    [16, -16],
    [1, -15],
])

frame_coordinates = np.array([
    [578, 53],
    [751, 325],
    [468, 463],
    [200, 345],
])


# pred_x2 = 11
# pred_y2 = -4

pt_1 = (443, 225)
pt_2 = (505, 444)

img = cv2.imread('paper_calibration/paper_calib2.jpg')
img = cv2.resize(img, (960, 960))
# cv2.imshow('1', img)
# cv2.waitKey(0)

homography_matrix, _ = cv2.findHomography(srcPoints=points_coordinates, dstPoints=frame_coordinates)
inv_homography_matrix = np.linalg.inv(homography_matrix)

# cv2.circle(img, frame_coordinates[0], radius=5, color=(255, 255, 0), thickness=-1)
# cv2.circle(img, frame_coordinates[1], radius=5, color=(255, 255, 0), thickness=-1)
# cv2.circle(img, frame_coordinates[2], radius=5, color=(255, 255, 0), thickness=-1)
# cv2.circle(img, frame_coordinates[3], radius=5, color=(255, 255, 0), thickness=-1)

mapped_point = np.dot(inv_homography_matrix, np.array([pred_x1, pred_y1, 1]))
mapped_point = (mapped_point / mapped_point[2])[:2]
mapped_point_int = tuple(np.round(mapped_point).astype(int))
# cv2.circle(img, mapped_point_int, radius=5, color=(0, 0, 255), thickness=-1)
print(mapped_point_int)

# mapped_point = np.dot(homography_matrix, np.array([pred_x2, pred_y2, 1]))
# mapped_point = (mapped_point / mapped_point[2])[:2]
# mapped_point_int = tuple(np.round(mapped_point).astype(int))
# cv2.circle(img, mapped_point_int, radius=5, color=(0, 0, 255), thickness=-1)
# print(mapped_point_int)

# mapped_point = np.dot(inv_homography_matrix, np.array([*pt_1, 1]))
# mapped_point = (mapped_point / mapped_point[2])[:2]
# mapped_point_int = tuple(mapped_point.astype(float))
# print(mapped_point_int)
# a = mapped_point_int
#
# mapped_point = np.dot(inv_homography_matrix, np.array([*pt_2, 1]))
# mapped_point = (mapped_point / mapped_point[2])[:2]
# mapped_point_int = tuple(mapped_point.astype(float))
# print(mapped_point_int)
# b = mapped_point_int
#
# distance = math.sqrt((a[0] - b[0])**2 + (b[1] - a[1])**2)
# print(distance)

# cv2.imshow("Frame with Mapped Point", img)
# cv2.waitKey(0)
