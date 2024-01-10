import os
import datetime
import csv
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import natsort


def get_timestamp(frame):
    timestamp = int(os.path.splitext(frame)[0]) / 1000
    adjusted_timestamp = timestamp - (3 * 60 * 60)  # Subtracting 3 hours
    dt_object = datetime.datetime.fromtimestamp(adjusted_timestamp)
    formatted_result = dt_object.strftime("%H%M%S.%f")[:-4]
    return formatted_result


class Homography:
    def __init__(self, project_name, prediction_folder, first_calib_lane_frames, second_calib_lane_frames,
                 imgsz=(920, 920), ):
        self.data_path = project_name
        self.calib_data_path = os.path.join(self.data_path, 'calibration')
        self.calib_lanes_frames = first_calib_lane_frames, second_calib_lane_frames
        self.calib_info = defaultdict(dict)
        self.calib_folders = None
        self.width, self.height = imgsz
        self.prediction_folder = prediction_folder
        self.clicked_plate_coords = []
        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.grid_spacing = 50
        self.pred_frames = None
        self.org_frame = None
        self.multi_point = True
        self.plate_coords = np.array([
            [76, 134],
            [319, 526],
            [233, 123],
            [904, 488],
        ])
        if self.multi_point:
            self.plate_coords = np.array([[103, 241],
                                          [348, 788],
                                          [440, 223],
                                          [918, 478]])

    def find_rectangle_corners(self, line1_coords, line2_coords):
        # Assuming line1_coords and line2_coords are lists of tuples (latitude, longitude)

        # Find intersection points
        intersection_point1 = line1_coords[0]
        intersection_point2 = line1_coords[-1]

        # Determine rectangle orientation
        is_vertical = abs(line1_coords[0][0] - line1_coords[1][0]) > abs(line1_coords[0][1] - line1_coords[1][1])

        # Calculate other two corners based on nearest points
        if is_vertical:
            corner3 = min(line2_coords, key=lambda point: abs(point[0] - intersection_point1[0]))
            corner4 = min(line2_coords, key=lambda point: abs(point[0] - intersection_point2[0]))
        else:
            corner3 = min(line2_coords, key=lambda point: abs(point[1] - intersection_point1[1]))
            corner4 = min(line2_coords, key=lambda point: abs(point[1] - intersection_point2[1]))

        return [intersection_point1, intersection_point2, corner3, corner4]

    def calibrate_frames(self):
        all_items = os.listdir(self.calib_data_path)
        self.calib_folders = [item for item in all_items if os.path.isdir(os.path.join(self.calib_data_path, item))]

        for calib_lane_frames, folder in zip(self.calib_lanes_frames, self.calib_folders):
            files = os.listdir(os.path.join(self.calib_data_path, folder))
            files = natsort.natsorted(files)

            if calib_lane_frames[0] < 1 or calib_lane_frames[1] > len(files) or calib_lane_frames[0] > \
                    calib_lane_frames[1]:
                raise ValueError("Invalid calib frame frame index Or GPS file is empty")

            frames = files[calib_lane_frames[0] - 1:calib_lane_frames[1]]
            filtered_frames = [filename for filename in frames if filename.endswith('50.jpg')]

            self.calib_info[folder]['start_finish_frames'] = filtered_frames

    def filter_coords(self, rect_coords):
        for coord in self.calib_info[self.calib_folders[0]]['start_finish_gps_coords']:
            if coord != rect_coords[0]:
                self.calib_info[self.calib_folders[0]]['start_finish_gps_coords'].remove(coord)
                self.calib_info[self.calib_folders[0]]['start_finish_frames'].pop(0)
            else:
                break

        for coord in reversed(self.calib_info[self.calib_folders[0]]['start_finish_gps_coords']):
            if coord != rect_coords[1]:
                self.calib_info[self.calib_folders[0]]['start_finish_gps_coords'].remove(coord)

                self.calib_info[self.calib_folders[0]]['start_finish_frames'].pop()
            else:
                break

        for coord in self.calib_info[self.calib_folders[1]]['start_finish_gps_coords']:
            if coord != rect_coords[2]:
                self.calib_info[self.calib_folders[1]]['start_finish_gps_coords'].remove(coord)
                self.calib_info[self.calib_folders[1]]['start_finish_frames'].pop(0)
            else:
                break

        for coord in reversed(self.calib_info[self.calib_folders[1]]['start_finish_gps_coords']):
            if coord != rect_coords[3]:
                self.calib_info[self.calib_folders[1]]['start_finish_gps_coords'].remove(coord)

                self.calib_info[self.calib_folders[1]]['start_finish_frames'].pop()
            else:
                break

    def calibrate_GPS(self):
        for calib_folder, info in self.calib_info.items():
            first_frame_time = get_timestamp(info['start_finish_frames'][0])
            last_frame_time = get_timestamp(info['start_finish_frames'][-1])

            gps_file = os.path.join(self.calib_data_path, calib_folder + '.txt')
            with open(gps_file, 'r') as input_file:
                for line in input_file:
                    data = line.split()
                    time, lat, long = data[0], float(data[1]), float(data[2])
                    # time, lat, long = data[1], float(data[2]), float(data[3])

                    if first_frame_time <= time <= last_frame_time:
                        if 'start_finish_gps_coords' not in self.calib_info[calib_folder]:
                            self.calib_info[calib_folder]['start_finish_gps_coords'] = []

                        self.calib_info[calib_folder]['start_finish_gps_coords'].append([lat, long])

        rectangle_corners = self.find_rectangle_corners(
            self.calib_info[self.calib_folders[0]]['start_finish_gps_coords'],
            self.calib_info[self.calib_folders[1]]['start_finish_gps_coords'])
        print(rectangle_corners)
        # self.filter_coords(rectangle_corners)

    def load_frames(self):
        if self.prediction_folder is None:
            all_folders = os.listdir(self.data_path)
            lane_folders = [item for item in all_folders if
                            os.path.isdir(os.path.join(self.data_path, item)) and item != 'calibration']
            lane_folder = os.path.join(self.data_path, lane_folders[0])
        else:
            lane_folder = os.path.join(self.data_path, self.prediction_folder)

        frames = []

        # Check if the folder exists
        if not os.path.exists(lane_folder):
            print(f"Folder path '{lane_folder}' does not exist.")
            return frames

        # Iterate through files in the folder
        for filename in natsort.natsorted(os.listdir(lane_folder)):
            filepath = os.path.join(lane_folder, filename)

            # Check if the file is an image (you can customize this check based on your file types)
            if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                if filename.endswith('50.jpg'):
                    continue
                # Read the image using OpenCV
                frame = cv2.imread(filepath)

                # Resize the frames to the cartrack size
                frame = cv2.resize(frame, (self.width, self.height))

                # Append the frame to the array
                frames.append(frame)

        return frames

    def plot_plate_coords(self, ):
        if self.multi_point:
            frame = self.pred_frames[0]
            for calibration_folder in self.calib_folders:
                with open(os.path.join(self.calib_data_path, calibration_folder, 'output_coords.txt')) as f:
                    for line in f:
                        cv2.circle(frame, tuple(eval(line))[0], radius=5, color=(255, 255, 255), thickness=-1)
        # cv2.imshow('3', frame)
        # cv2.waitKey(0)

    def perform_homography(self):
        source_gps_cords = []
        for calib_folder, info in self.calib_info.items():
            source_gps_cords.append(list(map(float, info['start_finish_gps_coords'][0])))
            source_gps_cords.append(list(map(float, info['start_finish_gps_coords'][-1])))
        source_gps_cords = np.array(source_gps_cords)
        self.homography_matrix, _ = cv2.findHomography(source_gps_cords, self.plate_coords, cv2.RANSAC, 5.0)
        self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)

        # Multi calibration point method:
        if self.multi_point:
            plate_coords = defaultdict(list)
            for calibration_folder in self.calib_folders:
                with open(os.path.join(self.calib_data_path, calibration_folder, 'output_coords.txt')) as f:
                    for line in f:
                        plate_coords[calibration_folder].append(list(eval(line))[0])
                plate_coords[calibration_folder] = np.array(plate_coords[calibration_folder])

            source_gps_cords = defaultdict(list)
            for calibration_folder, info in self.calib_info.items():
                for i in info['start_finish_gps_coords']:
                    source_gps_cords[calibration_folder].append(tuple(i))
                source_gps_cords[calibration_folder] = np.array(source_gps_cords[calibration_folder])

            # Trim lists for having same amount of elements
            for plate, gps in zip(plate_coords.items(), source_gps_cords.items()):
                distance = (len(plate[1]) - 1) // (len(gps[1]))
                plate_coords[plate[0]] = [plate[1][i] for i in range(len(plate[1])) if (i + 1) % (distance + 1) != 0]

            plate_coords = [item for sublist in zip(plate_coords['pass1'], plate_coords['pass2']) for item in sublist]
            plate_coords = np.array(plate_coords)
            source_gps_cords = [item for sublist in zip(source_gps_cords['pass1'], source_gps_cords['pass2']) for item in sublist]
            source_gps_cords = np.array(source_gps_cords)

            # source_gps_cords = source_gps_cords[:min_length]
            # plate_coords = plate_coords[:min_length]
            # self.homography_matrix, _ = cv2.findHomography(source_gps_cords, plate_coords, cv2.USAC_PROSAC, 500.0)
            # self.homography_matrix, _ = cv2.findHomography(source_gps_cords, plate_coords, cv2.LMEDS, 500.0)
            self.homography_matrix, _ = cv2.findHomography(source_gps_cords, plate_coords)

            self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
            # End

        self.pred_frames = self.load_frames()
        self.plot_plate_coords()
        prediction_gps_path = os.path.join(self.data_path, self.prediction_folder + '.txt')
        gps_xs = []
        gps_ys = []
        with open(prediction_gps_path, 'r') as input_file:
            for line in input_file:
                data = line.split()
                lat, long = float(data[1]), float(data[2])
                gps_xs.append(lat)
                gps_ys.append(long)

        if self.multi_point:
            # Shift track
            x1, y1 = gps_xs[0], gps_ys[0]
            x2, y2 = gps_xs[-1], gps_ys[-1]

            if x2 - x1 == 0:
                raise ValueError("Vertical line, cannot determine slope.")

            slope = (y2 - y1) / (x2 - x1)

            # Shift the coordinates
            shifted_coordinates = []
            direction = 'up'
            step = 0.019
            shifted_xs = []
            shifted_ys = []
            for x, y in zip(gps_xs, gps_ys):
                if direction == 'up':
                    y_shifted = y + step
                else:  # direction == 'down'
                    y_shifted = y - step

                # Shift along the slope
                x_shifted = x + step / slope

                shifted_xs.append(x_shifted)
                shifted_ys.append(y_shifted)
                shifted_coordinates.append((x_shifted, y_shifted))
            # end

        frame = self.pred_frames[0]
        for calib_folder, info in self.calib_info.items():
            for coord in info['start_finish_gps_coords']:
                x, y = map(float, coord)
                mapped_point = np.dot(self.homography_matrix, np.array([x, y, 1]))
                mapped_point = (mapped_point / mapped_point[2])[:2]
                mapped_point_int = tuple(mapped_point.astype(int))
                frame = cv2.circle(frame, mapped_point_int, radius=5, color=(0, 0, 255), thickness=-1)  # Fourth - Black
        cv2.imshow("Frame with Mapped Point", frame)
        cv2.waitKey(0)

        pred_track = []

        for gps_x, gps_y, frame in zip(gps_xs, gps_ys, self.pred_frames):
            # for gps_x, gps_y, frame in zip(shifted_xs, shifted_ys, self.pred_frames):
            frame = frame.copy()
            warped_frame = cv2.warpPerspective(frame, self.homography_matrix, (self.width, self.height))

            # Method 2 for finding the point
            # new_gps_coords = np.array([gps_x, gps_y]).reshape(-1, 1, 2)
            # transformed_additional_points = cv2.perspectiveTransform(new_gps_coords, self.homography_matrix)
            # # transformed_additional_points = cv2.perspectiveTransform((gps_x[0], gps_y[0]), homography_matrix)
            # for point in transformed_additional_points:
            #     cv2.circle(frame, tuple(map(int, tuple(point[0]))), 5, (0, 0, 255), -1)

            mapped_point = np.dot(self.homography_matrix, np.array([gps_x, gps_y, 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            mapped_point_int = tuple(np.round(mapped_point).astype(int))
            pred_track.append(mapped_point_int)

            cv2.circle(frame, self.plate_coords[0], radius=5, color=(0, 0, 255), thickness=-1)  # First - Red
            cv2.circle(frame, self.plate_coords[1], radius=5, color=(0, 255, 0), thickness=-1)  # Second - Green
            cv2.circle(frame, self.plate_coords[2], radius=5, color=(255, 0, 0), thickness=-1)  # Third - Blue
            cv2.circle(frame, self.plate_coords[3], radius=5, color=(0, 0, 0), thickness=-1)  # Fourth - Black

            cv2.circle(frame, mapped_point_int, radius=5, color=(0, 255, 0), thickness=-1)

            cv2.imshow("Frame with Mapped Point", frame)
            cv2.waitKey(250)
        with open('/home/user/Documents/accurate_speed_calculation/gps/data5/pass3/output_coords.txt') as f:
            for line in f:
                cv2.circle(frame, eval(line)[0], radius=5, color=(255, 255, 0), thickness=-1)
        for i in pred_track:
            cv2.circle(frame, tuple(i), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.imshow("Frame with Mapped Point", frame)
        cv2.waitKey(0)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.plate_coords.append((x, y))

    def plate_clib_open_frame(self, frame_path):
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (self.height, self.width))
        cv2.imshow("Calibrate Plate", frame)

        cv2.setMouseCallback("Calibrate Plate", self.mouse_callback, self.plate_coords)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                break
            elif key == 27:  # Esc key
                cv2.destroyAllWindows()
                return None

        cv2.setMouseCallback("Calibrate Plate", lambda *args: None)  # Disable mouse callback
        cv2.destroyAllWindows()

    def calibrate_plate(self):
        self.plate_coords = []

        for calib_folder, info in self.calib_info.items():
            first_frame_path = os.path.join(self.calib_data_path, calib_folder, info['start_finish_frames'][0])
            first_plate_coords = self.plate_clib_open_frame(first_frame_path)
            if first_plate_coords is not None:
                self.plate_coords.extend(first_plate_coords)

            last_frame_path = os.path.join(self.calib_data_path, calib_folder, info['start_finish_frames'][-1])
            last_plate_coords = self.plate_clib_open_frame(last_frame_path)
            if last_plate_coords is not None:
                self.plate_coords.extend(last_plate_coords)

        self.plate_coords = np.array(self.plate_coords)
        print(self.plate_coords)

    def plot_gps(self):
        for calib_folder, info in self.calib_info.items():
            latitudes, longitudes = zip(*info['start_finish_gps_coords'])
            plt.plot(longitudes, latitudes, label=calib_folder)
            plt.scatter(longitudes, latitudes)
        plt.title('GPS Coordinates Trajectory')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_homo_grid(self):
        # Create a grid on the frame
        grid_frame = self.pred_frames[0]
        # Create a grid on the frame
        for i in range(0, self.height, self.grid_spacing):
            cv2.line(grid_frame, (i, 0), (i, self.width), (0, 255, 0), 1)
            cv2.line(grid_frame, (0, i), (self.height, i), (0, 255, 0), 1)

        # Apply homography to the grid points
        mapped_points = cv2.perspectiveTransform(np.array(
            [[[i, j]] for j in range(0, self.width, self.grid_spacing) for i in
             range(0, self.height, self.grid_spacing)],
            dtype=np.float32), self.inv_homography_matrix)

        # Draw the mapped grid on the frame
        for point in mapped_points:
            cv2.circle(grid_frame, (int(point[0, 0]), int(point[0, 1])), 3, (255, 0, 0), -1)

        # Draw the mapped points on a plot
        plt.plot(mapped_points[:, 0, 0], mapped_points[:, 0, 1], 'bo', label='Mapped Grid Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Mapped Grid Points using Homography')
        plt.legend()
        plt.show()

        # Display the frame with grid
        cv2.imshow('Grid on Frame', grid_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #
        # grid_x, grid_y = np.meshgrid(
        #     np.arange(0, self.height, self.grid_spacing),
        #     np.arange(0, self.width, self.grid_spacing)
        # )
        #
        # # Stack the grid points to form homogeneous coordinates
        # grid_points = np.vstack((grid_x.flatten(), grid_y.flatten(), np.ones_like(grid_x.flatten())))
        #
        # # Apply homography to the grid points
        # mapped_points = np.dot(self.homography_matrix, grid_points)
        #
        # # Normalize homogeneous coordinates
        # mapped_points_normalized = mapped_points[:2, :] / mapped_points[2, :]
        #
        # # Plot the original grid
        # plt.plot(grid_x, grid_y, 'ro', label='Original Grid')
        #
        # # Plot the mapped grid
        # plt.plot(mapped_points_normalized[0, :], mapped_points_normalized[1, :], 'bo', label='Mapped Grid')
        #
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Grid Mapping using Homography')
        # plt.legend()
        # plt.show()

    def create_meshgrid(self, grid_resolution=10):
        coord1 = self.calib_info[list(self.calib_info.keys())[0]]['start_finish_gps_coords'][0]
        coord2 = self.calib_info[list(self.calib_info.keys())[0]]['start_finish_gps_coords'][-1]

        coord3 = self.calib_info[list(self.calib_info.keys())[1]]['start_finish_gps_coords'][0]
        coord4 = self.calib_info[list(self.calib_info.keys())[1]]['start_finish_gps_coords'][-1]

        n = 16

        print(coord1, coord2)

        x1, y1 = self.generate_parallel_lines(coord1, coord2)
        x, y = self.generate_parallel_lines2(coord1, coord2, coord3, coord4)

        # x = np.linspace(min(coord1[0], coord2[0]), max(coord1[0], coord2[0]) + (np.ptp([coord1[0], coord2[0]]) * 2), n)
        # y = np.linspace(min(coord1[1], coord2[1]), max(coord1[1], coord2[1]) + (np.ptp([coord1[1], coord2[1]]) * 2), n)
        # x = np.linspace(min(coord1[0], coord2[0]) - (np.ptp([coord1[0], coord2[0]]) * 3),
        #                 max(coord1[0], coord2[0]) + (np.ptp([coord1[0], coord2[0]]) * 3), n)
        # y = np.linspace(min(coord1[1], coord2[1]) - (np.ptp([coord1[0], coord2[0]]) * 3),
        #                 max(coord1[1], coord2[1]) + (np.ptp([coord1[1], coord2[1]]) * 3), n)

        # x = np.linspace(min(coord1[0], coord2[0])  )

        # for i in x:
        #     for j in y:
        #         plt.plot(i, j, marker='o', color='purple', markersize=10, )

        # plt.plot([x[0], x[3]], [y[3], y[0]], marker='o', color='purple', markersize=10, )
        # plt.plot([x[0], x[3]], [y[0], y[3]], marker='o', color='purple', markersize=10, )
        #
        # plt.plot([x[1], x[3]], [y[0], y[2]], marker='o', color='blue', markersize=10, )
        # plt.plot([x[2], x[3]], [y[0], y[1]], marker='o', color='blue', markersize=10, )
        #
        # plt.plot([x[0], x[2]], [y[1], y[3]], marker='o', color='brown', markersize=10, )
        # plt.plot([x[0], x[1]], [y[2], y[3]], marker='o', color='brown', markersize=10, )
        #
        # plt.plot([x[1], x[3]], [y[3], y[1]], marker='o', color='green', markersize=10, )
        # plt.plot([x[2], x[3]], [y[3], y[2]], marker='o', color='green', markersize=10, )
        #
        # plt.plot([x[0], x[2]], [y[2], y[0]], marker='o', color='black', markersize=10, )
        # plt.plot([x[0], x[1]], [y[1], y[0]], marker='o', color='black', markersize=10, )
        #
        # plt.plot([coord1[0], coord2[0]], [coord1[1], coord2[1]], marker='o', color='red', label='GPS Trajectory')
        #
        # np.set_printoptions(linewidth=np.inf)
        # # print(x)
        # # print(y)
        # # print(np.array([coord1[0], coord2[0], coord1[1], coord2[1]]))
        #
        # plt.legend()
        # plt.show()

        frame = self.pred_frames[0]

        mapped_point = np.dot(self.homography_matrix, np.array([*(coord1[0], coord1[1]), 1]))
        mapped_point = (mapped_point / mapped_point[2])[:2]
        mapped_point_int1 = tuple(mapped_point.astype(int))
        mapped_point = np.dot(self.homography_matrix, np.array([*(coord2[0], coord2[1]), 1]))
        mapped_point = (mapped_point / mapped_point[2])[:2]
        mapped_point_int2 = tuple(mapped_point.astype(int))
        cv2.line(frame, mapped_point_int1, mapped_point_int2, color=(0, 255, 255), thickness=1)

        for i in range(0, len(x) - 1, 2):
            mapped_point = np.dot(self.homography_matrix, np.array([*(x[i], y[i]), 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            mapped_point_int1 = tuple(mapped_point.astype(int))

            mapped_point = np.dot(self.homography_matrix, np.array([*(x[i + 1], y[i + 1]), 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            mapped_point_int2 = tuple(mapped_point.astype(int))

            cv2.line(frame, mapped_point_int1, mapped_point_int2, color=(0, 255, 255), thickness=1)

        # for i in x:
        #     for j in y:
        #         # mapped_point = np.dot(self.homography_matrix, np.array([*(i, j), 1]))
        #         # mapped_point = (mapped_point / mapped_point[2])[:2]
        #         # mapped_point_int = tuple(mapped_point.astype(int))
        #         # cv2.circle(frame, mapped_point_int, radius=2, color=(255, 255, 0), thickness=-1)  # Third - Blue
        #         # if coord1 == (i, j) or coord2 == (i, j):
        #         #     cv2.circle(frame, mapped_point_int, radius=3, color=(0, 255, 0), thickness=-1)  # Third - Blue
        #
        #         mapped_point = np.dot(self.homography_matrix, np.array([*(i, j), 1]))
        #         mapped_point = (mapped_point / mapped_point[2])[:2]
        #         mapped_point_int1 = tuple(mapped_point.astype(int))
        #         i = next(i)
        #         j = next(j)
        #         mapped_point = np.dot(self.homography_matrix, np.array([*(i, j), 1]))
        #         mapped_point = (mapped_point / mapped_point[2])[:2]
        #         mapped_point_int2 = tuple(mapped_point.astype(int))
        #         cv2.line(frame, mapped_point_int1, mapped_point_int2, color=(0, 255, 255), thickness=1)

        cv2.imshow('a', frame)
        cv2.waitKey(0)

    def generate_parallel_lines(self, coord1, coord2):
        # Given coordinates
        x1, y1 = coord1
        x2, y2 = coord2

        # Calculate the slope of the line
        m = (y2 - y1) / (x2 - x1)

        # Calculate the perpendicular slope
        perpendicular_m = -1 / m

        # Calculate the length of the line segment
        line_length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        # Number of lines to create
        n = 13  # You can change this value to create a different number of lines

        # Plot the original line
        plt.plot([x1, x2], [y1, y2], label="Original Line")

        xs = []
        ys = []
        # Loop to create n parallel lines
        for i in range(1, n + 1):
            offset = 0.00001 * i  # Adjust the offset based on the iteration
            # offset = 0.000009 * i  # Adjust the offset based on the iteration
            dx = offset / (line_length / 2)
            dy = perpendicular_m * dx / 2.8

            left_x1 = x1 + dx
            left_y1 = y1 + dy
            left_x2 = x2 + dx
            left_y2 = y2 + dy

            right_x1 = x1 - dx
            right_y1 = y1 - dy
            right_x2 = x2 - dx
            right_y2 = y2 - dy

            # Print the beginning and endings of the generated lines
            print(f"Parallel Line {i} to the Left: ({left_x1}, {left_y1}) to ({left_x2}, {left_y2})")
            print(f"Parallel Line {i} to the Right: ({right_x1}, {right_y1}) to ({right_x2}, {right_y2})")
            xs.append(left_x1)
            xs.append(left_x2)
            xs.append(right_x1)
            xs.append(right_x2)

            ys.append(left_y1)
            ys.append(left_y2)
            ys.append(right_y1)
            ys.append(right_y2)

            # Plot the parallel lines
            plt.plot([left_x1, left_x2], [left_y1, left_y2], label=f"Parallel Line {i} to the Left")
            plt.plot([right_x1, right_x2], [right_y1, right_y2], label=f"Parallel Line {i} to the Right")

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
        return xs, ys

    def generate_parallel_lines2(self, coord1, coord2, coord3, coord4):
        n = 5
        upper_line = np.linspace(coord1, coord3, n)
        lower_line = np.linspace(coord2, coord4, n)

        for c1, c2 in zip(upper_line, lower_line):
            plt.plot([c1[0], c2[0]], [c1[1], c2[1]], label=f"1")

        xs = np.empty(len(upper_line) * 2)
        xs[0::2] = upper_line[:, 0]  # Alternately assign upper line x values
        xs[1::2] = lower_line[:, 0]  # Alternately assign lower line x values

        ys = np.empty(len(upper_line) * 2)
        ys[0::2] = upper_line[:, 1]  # Alternately assign upper line y values
        ys[1::2] = lower_line[:, 1]  # Alternately assign lower line y values

        # plt.plot(xs, ys)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
        return xs, ys

    def shift_track(self, direction, step):
        # plot it:
        for calib_folder, info in self.calib_info.items():
            latitudes, longitudes = zip(*info['start_finish_gps_coords'])
            # plt.plot(longitudes, latitudes, label=calib_folder, alpha=0.5)
            plt.scatter(longitudes, latitudes, label=calib_folder)

        coordinates = self.calib_info['pass1']['start_finish_gps_coords']
        if direction not in ['up', 'down']:
            raise ValueError("Direction must be 'up' or 'down'.")

        if len(coordinates) < 2:
            raise ValueError("At least two coordinates are required.")

        # Calculate the slope of the line connecting the first and last coordinates
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[-1]

        if x2 - x1 == 0:
            raise ValueError("Vertical line, cannot determine slope.")

        slope = (y2 - y1) / (x2 - x1)

        # Shift the coordinates
        shifted_coordinates = []
        for x, y in coordinates:
            if direction == 'up':
                y_shifted = y + step
            else:  # direction == 'down'
                y_shifted = y - step

            # Shift along the slope
            x_shifted = x + step / slope

            shifted_coordinates.append((x_shifted, y_shifted))

        self.calib_info['pass1']['start_finish_gps_coords'] = shifted_coordinates

        latitudes, longitudes = zip(*self.calib_info['pass1']['start_finish_gps_coords'])
        plt.plot(longitudes, latitudes, label='pass 1 shifted', color='red', linewidth=6, alpha=0.75)
        # plt.scatter(longitudes, latitudes, color='red', label='pass 1 shifted', alpha=0.6)
        plt.title('GPS Coordinates Trajectory')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.show()
        return shifted_coordinates

    def run(self):
        self.calibrate_frames()
        self.calibrate_GPS()
        if self.multi_point:
            self.shift_track('up', 0.019)
        # self.shift_track('up', 0.003)
        # self.plot_gps()
        # self.calibrate_plate()
        self.perform_homography()
        # self.draw_homo_grid()
        self.create_meshgrid()


if __name__ == "__main__":
    project_name = 'data5'
    prediction_folder = 'pass3'
    # first_calib_lane = (230, 295)
    # second_calib_lane = (237, 306)

    # first_calib_lane = (59, 191)
    # second_calib_lane = (17, 172)

    first_calib_lane = (3, 87)
    second_calib_lane = (16, 83)

    model = Homography(project_name, prediction_folder, first_calib_lane, second_calib_lane)
    model.run()
