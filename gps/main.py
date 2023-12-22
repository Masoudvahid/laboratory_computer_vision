import os
import datetime
import csv
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import natsort


class Homography:
    def __init__(self, first_calib_lane_frames, second_calib_lane_frames, ):
        self.data_path = 'data'
        self.calib_data_path = os.path.join(self.data_path, 'calibration')
        self.calib_lanes_frames = first_calib_lane_frames, second_calib_lane_frames
        self.calib_info = defaultdict(dict)
        self.width, self.height = 920, 920
        self.calib_folders = None

        # self.plate_coords = np.array([[618, 24],
        #                               [102, 454],
        #                               [870, 27],
        #                               [671, 465]])
        self.plate_coords = np.array([[559, 19],
                                      [47, 412],
                                      [835, 21],
                                      [664, 428]])
        # self.plate_coords = np.array([
        #     [76, 230],
        #     [271, 776],
        #     [412, 212],
        #     [888, 464],
        # ])

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

            # if calib_lane_frames[0] < 1 or calib_lane_frames[1] > len(files) or calib_lane_frames[0] > \
            #         calib_lane_frames[1]:
            #     raise ValueError("Invalid calib frame frame index Or GPS file is empty")

            self.calib_info[folder]['start_finish_frames'] = files[calib_lane_frames[0] - 1:calib_lane_frames[1]]

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
            first_frame_time = self.get_timestamp(info['start_finish_frames'][0])
            last_frame_time = self.get_timestamp(info['start_finish_frames'][-1])

            gps_file = os.path.join(self.calib_data_path, calib_folder + '.txt')
            with open(gps_file, 'r') as input_file:
                for line in input_file:
                    data = line.split()
                    time, lat, long = data[1], float(data[2]), float(data[3])

                    if first_frame_time <= time <= last_frame_time:
                        if 'start_finish_gps_coords' not in self.calib_info[calib_folder]:
                            self.calib_info[calib_folder]['start_finish_gps_coords'] = []

                        self.calib_info[calib_folder]['start_finish_gps_coords'].append([lat, long])

        rectangle_corners = self.find_rectangle_corners(
            self.calib_info[self.calib_folders[0]]['start_finish_gps_coords'],
            self.calib_info[self.calib_folders[1]]['start_finish_gps_coords'])
        print(rectangle_corners)
        self.filter_coords(rectangle_corners)

    def get_timestamp(self, frame):
        timestamp = int(os.path.splitext(frame)[0]) / 1000
        adjusted_timestamp = timestamp - (3 * 60 * 60)  # Subtracting 3 hours
        dt_object = datetime.datetime.fromtimestamp(adjusted_timestamp)
        formatted_result = dt_object.strftime("%H%M%S.%f")[:-4]
        return formatted_result

    def load_frames(self):
        all_folders = os.listdir(self.data_path)
        lane_folders = [item for item in all_folders if
                        os.path.isdir(os.path.join(self.data_path, item)) and item != 'calibration']

        lane_folder = os.path.join(self.data_path, lane_folders[0])

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
        homography_matrix, _ = cv2.findHomography(srcPoints=source_gps_cords, dstPoints=self.plate_coords)

        frames = self.load_frames()
        gps_file_path = os.path.join(self.data_path, 'lane2.csv')
        gps_data = pd.read_csv(gps_file_path)
        gps_xs = gps_data.iloc[:, 1].values.reshape(-1, 1)  # Assuming the second column is latitude
        gps_ys = gps_data.iloc[:, 2].values.reshape(-1, 1)  # Assuming the third column is longitude
        print(len(gps_xs), len(frames))

        for gps_x, gps_y, frame in zip(gps_xs, gps_ys, frames):
            warped_frame = cv2.warpPerspective(frame, homography_matrix, (self.width, self.height))

            # Method 2 for finding the point
            new_gps_coords = np.array([gps_x, gps_y]).reshape(-1, 1, 2)
            transformed_additional_points = cv2.perspectiveTransform(new_gps_coords, homography_matrix)
            # transformed_additional_points = cv2.perspectiveTransform((gps_x[0], gps_y[0]), homography_matrix)
            for point in transformed_additional_points:
                cv2.circle(frame, tuple(map(int, tuple(point[0]))), 5, (0, 0, 255), -1)

            gps_x = gps_x[0]
            gps_y = gps_y[0]

            mapped_point = np.dot(homography_matrix, np.array([gps_x, gps_y, 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            mapped_point_int = tuple(np.round(mapped_point).astype(int))

            cv2.circle(frame, self.plate_coords[0], radius=5, color=(255, 255, 0), thickness=-1)
            cv2.circle(frame, self.plate_coords[1], radius=5, color=(255, 255, 0), thickness=-1)
            cv2.circle(frame, self.plate_coords[2], radius=5, color=(255, 255, 0), thickness=-1)
            cv2.circle(frame, self.plate_coords[3], radius=5, color=(255, 255, 0), thickness=-1)

            cv2.circle(frame, mapped_point_int, radius=5, color=(0, 255, 0), thickness=-1)

            cv2.imshow("Frame with Mapped Point", frame)
            cv2.waitKey(250)

    def plate_clib_open_frame(self, frame_path):
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (self.height, self.width))
        cv2.imshow("Calibrate Plate", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        user_input = input('Enter Plate Centers. Ex: 100 100:  ')
        input_list = [int(x) for x in user_input.split()]
        if len(input_list) == 2 and all(isinstance(x, int) for x in input_list):
            return input_list
        else:
            print("Please enter two integers separated by a space.")
            return None

    def calibrate_plate(self):
        plate_coords = []

        for calib_folder, info in self.calib_info.items():
            first_frame_path = os.path.join(self.calib_data_path, calib_folder, info['start_finish_frames'][0])
            first_plate_coords = self.plate_clib_open_frame(first_frame_path)
            if first_plate_coords is not None:
                plate_coords.append(first_plate_coords)

            last_frame_path = os.path.join(self.calib_data_path, calib_folder, info['start_finish_frames'][-1])
            last_plate_coords = self.plate_clib_open_frame(last_frame_path)
            if last_plate_coords is not None:
                plate_coords.append(last_plate_coords)

        self.plate_coords = np.array(plate_coords)

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

    def run(self):
        self.calibrate_frames()
        self.calibrate_GPS()
        self.plot_gps()
        # self.calibrate_plate()
        self.perform_homography()


if __name__ == "__main__":
    # first_calib_lane = (3, 87)
    # second_calib_lane = (20, 87)
    # second_calib_lane = (18, 101)

    first_calib_lane = (7, 64)
    second_calib_lane = (5, 51)
    model = Homography(first_calib_lane, second_calib_lane)
    model.run()
