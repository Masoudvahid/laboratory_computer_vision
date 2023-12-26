import cv2
import numpy as np
from scipy.interpolate import interp1d

points_coordinates = np.array([
    [0, 0],
    [16, 0],
    [16, -16],
    [1, -15],
])

frame_coordinates = np.array([
    [443, 225],
    [751, 325],
    [468, 463],
    [200, 345],
])

pred_x1 = 8
pred_y1 = -8

pred_x2 = 11
pred_y2 = -4

img = cv2.imread('paper_calibration/paper_calib2.jpg')
img = cv2.resize(img, (920, 920))
# cv2.imshow('1', img)
# cv2.waitKey(0)


# interpolate_x = interp1d(points_coordinates[:, 0], video_coordinates[:, 0], kind='linear', fill_value='extrapolate')
# interpolate_y = interp1d(points_coordinates[:, 1], video_coordinates[:, 1], kind='linear', fill_value='extrapolate')
# mapped_point_int = (int(interpolate_x(pred_x)), int(interpolate_y(pred_y)))

# matrix = cv2.getPerspectiveTransform(np.float32(frame_coordinates), np.float32(points_coordinates))
# test_point = np.array([[300, 300]], dtype=np.float32)
# mapped_point_int = cv2.perspectiveTransform(test_point.reshape(-1, 1, 2), matrix).astype(int)

homography_matrix, _ = cv2.findHomography(srcPoints=points_coordinates, dstPoints=frame_coordinates)
# mapped_point = np.dot(homography_matrix, np.array([pred_x1, pred_y1, 1]))
# mapped_point = (mapped_point / mapped_point[2])[:2]
# mapped_point_int = tuple(np.round(mapped_point).astype(int))

# mapped_point_int = cv2.perspectiveTransform(np.float32([-8, 8]).reshape(-1, 1, 2), homography_matrix).astype(int)

# calibration_points_homogeneous = np.column_stack((points_coordinates, np.ones(len(points_coordinates))))
# transformation_matrix = np.linalg.solve(calibration_points_homogeneous, frame_coordinates)
# mapped_point_int = np.dot(calibration_points_homogeneous, transformation_matrix.T)


cv2.circle(img, frame_coordinates[0], radius=5, color=(255, 255, 0), thickness=-1)
cv2.circle(img, frame_coordinates[1], radius=5, color=(255, 255, 0), thickness=-1)
cv2.circle(img, frame_coordinates[2], radius=5, color=(255, 255, 0), thickness=-1)
cv2.circle(img, frame_coordinates[3], radius=5, color=(255, 255, 0), thickness=-1)

mapped_point = np.dot(homography_matrix, np.array([pred_x1, pred_y1, 1]))
mapped_point = (mapped_point / mapped_point[2])[:2]
mapped_point_int = tuple(np.round(mapped_point).astype(int))
cv2.circle(img, mapped_point_int, radius=5, color=(0, 0, 255), thickness=-1)

mapped_point = np.dot(homography_matrix, np.array([pred_x2, pred_y2, 1]))
mapped_point = (mapped_point / mapped_point[2])[:2]
mapped_point_int = tuple(np.round(mapped_point).astype(int))
cv2.circle(img, mapped_point_int, radius=5, color=(0, 0, 255), thickness=-1)

# shape = (900, 900)
# im_fin = cv2.warpPerspective(img, homography_matrix, shape)

cv2.imshow("Frame with Mapped Point", img)
cv2.waitKey(0)
