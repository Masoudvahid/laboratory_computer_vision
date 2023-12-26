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
        self.plate_coords = np.array([
            [90, 4],
            [336, 855],
            [159, 4],
            [819, 836],
        ])

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

    def perform_homography(self):
        source_gps_cords = []
        for calib_folder, info in self.calib_info.items():
            source_gps_cords.append(list(map(float, info['start_finish_gps_coords'][0])))
            source_gps_cords.append(list(map(float, info['start_finish_gps_coords'][-1])))
        source_gps_cords = np.array(source_gps_cords)
        self.homography_matrix, _ = cv2.findHomography(srcPoints=source_gps_cords, dstPoints=self.plate_coords)
        self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)


        self.pred_frames = self.load_frames()
        prediction_gps_path = os.path.join(self.data_path, self.prediction_folder + '.txt')
        gps_xs = []
        gps_ys = []
        with open(prediction_gps_path, 'r') as input_file:
            for line in input_file:
                data = line.split()
                lat, long = float(data[1]), float(data[2])
                gps_xs.append(lat)
                gps_ys.append(long)

        for gps_x, gps_y, frame in zip(gps_xs, gps_ys, self.pred_frames):
            warped_frame = cv2.warpPerspective(frame, self.homography_matrix, (self.width, self.height))

            # Method 2 for finding the point
            new_gps_coords = np.array([gps_x, gps_y]).reshape(-1, 1, 2)
            transformed_additional_points = cv2.perspectiveTransform(new_gps_coords, self.homography_matrix)
            # transformed_additional_points = cv2.perspectiveTransform((gps_x[0], gps_y[0]), homography_matrix)
            for point in transformed_additional_points:
                cv2.circle(frame, tuple(map(int, tuple(point[0]))), 5, (0, 0, 255), -1)

            mapped_point = np.dot(self.homography_matrix, np.array([gps_x, gps_y, 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            mapped_point_int = tuple(np.round(mapped_point).astype(int))

            cv2.circle(frame, self.plate_coords[0], radius=5, color=(0, 0, 255), thickness=-1)  # First - Red
            cv2.circle(frame, self.plate_coords[1], radius=5, color=(0, 255, 0), thickness=-1)  # Second - Green
            cv2.circle(frame, self.plate_coords[2], radius=5, color=(255, 0, 0), thickness=-1)  # Third - Blue
            cv2.circle(frame, self.plate_coords[3], radius=5, color=(0, 0, 0), thickness=-1)  # Fourth - Black

            cv2.circle(frame, mapped_point_int, radius=5, color=(0, 255, 0), thickness=-1)

            # cv2.imshow("Frame with Mapped Point", frame)
            # cv2.waitKey(250)

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
            [[[i, j]] for j in range(0, self.width, self.grid_spacing) for i in range(0, self.height, self.grid_spacing)],
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

    def create_meshgrid(self, coords1, coords2, grid_resolution=0.0001):

        min_lat = min(min(coords1[:, 0]), min(coords2[:, 0]))
        max_lat = max(max(coords1[:, 0]), max(coords2[:, 0]))

        min_lon = min(min(coords1[:, 1]), min(coords2[:, 1]))
        max_lon = max(max(coords1[:, 1]), max(coords2[:, 1]))

        # Create a mesh grid
        lat_range = np.arange(min_lat, max_lat, grid_resolution)
        lon_range = np.arange(min_lon, max_lon, grid_resolution)

        meshgrid = np.meshgrid(lat_range, lon_range)

        return meshgrid

    def plot_meshgrid(self, meshgrid, elevations, title="Mesh Grid"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(meshgrid[0], meshgrid[1], elevations, cmap='viridis', edgecolor='k')
        ax.set_title(title)
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_zlabel('Elevation')

        plt.show()

    def run(self):
        self.calibrate_frames()
        self.calibrate_GPS()
        self.plot_gps()
        # self.calibrate_plate()
        self.perform_homography()
        self.draw_homo_grid()
        # self.create_meshgrid()


if __name__ == "__main__":
    project_name = 'data4'
    prediction_folder = 'pass6'
    # first_calib_lane = (230, 295)
    # second_calib_lane = (237, 306)
    first_calib_lane = (145, 299)
    second_calib_lane = (162, 316)

    model = Homography(project_name, prediction_folder, first_calib_lane, second_calib_lane)
    model.run()
