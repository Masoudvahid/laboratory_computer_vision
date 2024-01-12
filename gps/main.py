import os
import datetime
from collections import defaultdict
import cv2
import numpy as np
import matplotlib.pyplot as plt
import natsort


def get_timestamp(frame):
    timestamp = int(os.path.splitext(frame)[0]) / 1000
    adjusted_timestamp = timestamp - (3 * 60 * 60)  # Subtracting 3 hours
    dt_object = datetime.datetime.fromtimestamp(adjusted_timestamp)
    formatted_result = dt_object.strftime("%H%M%S.%f")[:-4]
    return float(formatted_result)

class Homography:
    def __init__(self, project_name, prediction_folder, first_calib_lane_frames, second_calib_lane_frames,
                 imgsz=(920, 920), ):
        self.data_path = project_name
        self.calib_data_path = os.path.join(self.data_path, 'calibration')
        self.calib_lanes_frames = first_calib_lane_frames, second_calib_lane_frames
        self.calib_dict = defaultdict(dict)
        self.calib_folders = None
        self.width, self.height = imgsz
        self.prediction_folder = prediction_folder
        self.clicked_plate_coords = []
        self.homography_matrix = None
        self.inv_homography_matrix = None
        self.pred_frames = None
        self.org_frame = None
        self.multi_point = False

        ##### P2 #######
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
        ####### P3 #######
        self.plate_coords = np.array([[123, 98],
                                      [240, 507],
                                      [362, 84],
                                      [827, 456]])

    def calibrate_frames(self):
        all_items = os.listdir(self.calib_data_path)
        self.calib_folders = [item for item in all_items if os.path.isdir(os.path.join(self.calib_data_path, item))]

        for calib_lane_frames, folder in zip(self.calib_lanes_frames, self.calib_folders):
            files = os.listdir(os.path.join(self.calib_data_path, folder))
            files = natsort.natsorted(files)

            if calib_lane_frames[0] < 1 or calib_lane_frames[1] > len(files) or calib_lane_frames[0] > calib_lane_frames[1]:
                raise ValueError("Invalid calib frame index Or GPS file is empty")

            frames = files[calib_lane_frames[0] - 1:calib_lane_frames[1]]

            frames_dict = {get_timestamp(key): None for key in frames}
            self.calib_dict[folder]['time_coord_dict'] = frames_dict

    def calibrate_GPS_with_frames(self):
        for calib_folder, calib_info in self.calib_dict.items():
            frames_time = [info for info in calib_info['time_coord_dict'].keys()]
            last_frame_time = frames_time[-1]

            # Find corresponding gps coordinates for each frame
            gps_file = os.path.join(self.calib_data_path, calib_folder + '.txt')
            with open(gps_file, 'r') as input_file:
                for line in input_file:
                    data = line.split()
                    time, lat, long = float(data[0]), float(data[1]), float(data[2])
                    if time in frames_time:
                        calib_info['time_coord_dict'][time] = [lat, long]
                    if time > last_frame_time:
                        break

            # Remove frames which don't have gps coordinates
            for time in frames_time:
                if calib_info['time_coord_dict'][time] is None:
                    del calib_info['time_coord_dict'][time]

    def load_frames(self, skip_frames: tuple):
        pred_folder_path = os.path.join(self.data_path, self.prediction_folder)
        frames = []

        # Check if the folder exists
        if not os.path.exists(pred_folder_path):
            print(f"Folder path '{pred_folder_path}' does not exist.")
            return frames

        # Iterate through files in the folder
        for filename in natsort.natsorted(os.listdir(pred_folder_path)):
            filepath = os.path.join(pred_folder_path, filename)

            # Check if the file is an image (you can customize this check based on your file types)
            if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                if filename.endswith(skip_frames):
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

    def generate_gps_coords(self, lane_plate_num):
        generated_lanes = defaultdict(list)
        for calib_folder, calib_info in self.calib_dict.items():
            first_coord = calib_info['time_coord_dict'][0]
            last_coord = calib_info['time_coord_dict'][-1]
            generated_lanes[calib_folder] = np.linspace(first_coord, last_coord, lane_plate_num[calib_folder])
        return generated_lanes

    def perform_homography(self):
        source_gps_cords = []
        for calib_folder, calib_info in self.calib_dict.items():
            source_gps_cords.append(list(calib_info['time_coord_dict'].values())[0])
            source_gps_cords.append(list(calib_info['time_coord_dict'].values())[-1])
        source_gps_cords = np.array(source_gps_cords)
        normalized_plate_coords = np.array([(x[0] / self.height, x[1] / self.width) for x in self.plate_coords])
        self.homography_matrix, _ = cv2.findHomography(source_gps_cords, normalized_plate_coords, cv2.RANSAC, 5.0)
        self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
        np.save('calib_4pt', self.homography_matrix)
        np.save('calib_4pt_inv', self.inv_homography_matrix)

        # Multi calibration point method:
        if self.multi_point:
            lane_plane_num = defaultdict(int)
            plate_coords = defaultdict(list)
            for calibration_folder in self.calib_folders:
                lane_plane_num[calibration_folder] = 0
                with open(os.path.join(self.calib_data_path, calibration_folder, 'output_coords.txt')) as f:
                    for line in f:
                        lane_plane_num[calibration_folder] += 1
                        plate_coords[calibration_folder].append(list(eval(line))[0])
                plate_coords[calibration_folder] = np.array(plate_coords[calibration_folder])

            source_gps_cords = defaultdict(list)
            for calibration_folder, calib_info in self.calib_dict.items():
                for i in calib_info['time_coord_dict']:
                    source_gps_cords[calibration_folder].append(tuple(i))
                source_gps_cords[calibration_folder] = np.array(source_gps_cords[calibration_folder])

            source_gps_cords = self.generate_gps_coords(lane_plane_num)

            plate_coords = np.concatenate((plate_coords[list(plate_coords.keys())[0]],
                                           plate_coords[list(plate_coords.keys())[1]]), axis=0)
            source_gps_cords = np.concatenate((source_gps_cords[list(source_gps_cords.keys())[0]],
                                               source_gps_cords[list(source_gps_cords.keys())[1]]), axis=0)

            # self.homography_matrix, _ = cv2.findHomography(source_gps_cords, plate_coords, cv2.USAC_PROSAC, 500.0)
            # self.homography_matrix, _ = cv2.findHomography(source_gps_cords, plate_coords, cv2.LMEDS, 500.0)
            self.homography_matrix, _ = cv2.findHomography(source_gps_cords, plate_coords)
            self.inv_homography_matrix = np.linalg.inv(self.homography_matrix)
            np.save('many_calib_pt', self.homography_matrix)
            np.save('many_calib_pt_inv', self.inv_homography_matrix)
            # End

        self.pred_frames = self.load_frames(skip_frames=('50.jpg',))
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
        for coord in source_gps_cords:
            x, y = map(float, coord)
            mapped_point = np.dot(self.homography_matrix, np.array([x, y, 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            unormalized_point = np.array([mapped_point[0] * self.height, mapped_point[1] * self.width])
            mapped_point_int = tuple(unormalized_point.astype(int))
            frame = cv2.circle(frame, mapped_point_int, radius=5, color=(0, 0, 255), thickness=-1)  # Fourth - Black
        cv2.imshow("Frame with Mapped Point", frame)
        cv2.waitKey(0)

        pred_track = []

        for gps_x, gps_y, frame in zip(gps_xs, gps_ys, self.pred_frames):
            # for gps_x, gps_y, frame in zip(shifted_xs, shifted_ys, self.pred_frames):
            frame = frame.copy()
            mapped_point = np.dot(self.homography_matrix, np.array([gps_x, gps_y, 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            unormalized_point = np.array([mapped_point[0] * self.height, mapped_point[1] * self.width])
            mapped_point_int = tuple(np.round(unormalized_point).astype(int))
            pred_track.append(mapped_point_int)

            cv2.circle(frame, self.plate_coords[0], radius=5, color=(0, 0, 255), thickness=-1)  # First - Red
            cv2.circle(frame, self.plate_coords[1], radius=5, color=(0, 255, 0), thickness=-1)  # Second - Green
            cv2.circle(frame, self.plate_coords[2], radius=5, color=(255, 0, 0), thickness=-1)  # Third - Blue
            cv2.circle(frame, self.plate_coords[3], radius=5, color=(0, 0, 0), thickness=-1)  # Fourth - Black

            cv2.circle(frame, mapped_point_int, radius=5, color=(0, 255, 0), thickness=-1)

            # cv2.imshow("Frame with Mapped Point", frame)
            # cv2.waitKey(250)

        # with open('/home/user/Documents/accurate_speed_calculation/gps/data5/pass3/output_coords.txt') as f:
        #     for line in f:
        #         cv2.circle(frame, eval(line)[0], radius=5, color=(255, 255, 0), thickness=-1)
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

        for calib_folder, calib_info in self.calib_dict.items():
            first_frame_path = os.path.join(self.calib_data_path, calib_folder, calib_info['frames_list'][0])
            first_plate_coords = self.plate_clib_open_frame(first_frame_path)
            if first_plate_coords is not None:
                self.plate_coords.extend(first_plate_coords)

            last_frame_path = os.path.join(self.calib_data_path, calib_folder, calib_info['frames_list'][-1])
            last_plate_coords = self.plate_clib_open_frame(last_frame_path)
            if last_plate_coords is not None:
                self.plate_coords.extend(last_plate_coords)

        self.plate_coords = np.array(self.plate_coords)
        print(self.plate_coords)

    def plot_gps(self):
        for calib_folder, calib_info in self.calib_dict.items():
            latitudes, longitudes = zip(*calib_info['time_coord_dict'].values())
            plt.plot(longitudes, latitudes, label=calib_folder)
            plt.scatter(longitudes, latitudes)
        plt.title('GPS Coordinates Trajectory')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    def create_meshgrid(self, grid_resolution=10):
        mesh_corners = []
        for calib_info in self.calib_dict.values():
            mesh_corners.append(list(calib_info['time_coord_dict'].values())[0])
            mesh_corners.append(list(calib_info['time_coord_dict'].values())[-1])

        x, y = self.generate_parallel_lines(*mesh_corners)

        frame = self.pred_frames[0]

        for i in range(0, len(x) - 1, 2):
            mapped_point = np.dot(self.homography_matrix, np.array([*(x[i], y[i]), 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            unormalized_point = np.array([mapped_point[0] * self.height, mapped_point[1] * self.width])

            mapped_point_int1 = tuple(unormalized_point.astype(int))

            mapped_point = np.dot(self.homography_matrix, np.array([*(x[i + 1], y[i + 1]), 1]))
            mapped_point = (mapped_point / mapped_point[2])[:2]
            unormalized_point = np.array([mapped_point[0] * self.height, mapped_point[1] * self.width])

            mapped_point_int2 = tuple(unormalized_point.astype(int))

            cv2.line(frame, mapped_point_int1, mapped_point_int2, color=(0, 255, 255), thickness=1)

        n = 15
        horizontal_lines1 = np.linspace(mesh_corners[0], mesh_corners[1], n)
        horizontal_lines2 = np.linspace(mesh_corners[2], mesh_corners[3], n)
        for l1, l2 in zip(horizontal_lines1, horizontal_lines2):
            mapped_l1 = np.dot(self.homography_matrix, np.array([*l1, 1]))
            mapped_l1 = (mapped_l1 / mapped_l1[2])[:2]
            unormalized_point = np.array([mapped_l1[0] * self.height, mapped_l1[1] * self.width])

            mapped_l1_int = tuple(np.round(unormalized_point).astype(int))

            mapped_l2 = np.dot(self.homography_matrix, np.array([*l2, 1]))
            mapped_l2 = (mapped_l2 / mapped_l2[2])[:2]
            unormalized_point = np.array([mapped_l2[0] * self.height, mapped_l2[1] * self.width])

            mapped_l2_int = tuple(np.round(unormalized_point).astype(int))

            cv2.line(frame, mapped_l1_int, mapped_l2_int, color=(0, 255, 255), thickness=1)

        cv2.imshow('perspective representation', frame)
        cv2.waitKey(0)

    def generate_parallel_lines(self, coord1, coord2, coord3, coord4):
        n = 8
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
        for calib_folder, info in self.calib_dict.items():
            latitudes, longitudes = zip(*info['time_coord_dict'])
            plt.scatter(longitudes, latitudes, label=calib_folder)

        coordinates = self.calib_dict['pass1']['time_coord_dict']
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

        self.calib_dict['pass1']['time_coord_dict'] = shifted_coordinates

        latitudes, longitudes = zip(*self.calib_dict['pass1']['time_coord_dict'])
        plt.plot(longitudes, latitudes, label='pass 1 shifted', color='red', linewidth=6, alpha=0.75)
        plt.title('GPS Coordinates Trajectory')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.show()
        return shifted_coordinates

    def run(self):
        self.calibrate_frames()
        self.calibrate_GPS_with_frames()
        if self.multi_point:
            self.shift_track('up', 0.019)
        # self.shift_track('up', 0.003)
        self.plot_gps()
        # self.calibrate_plate()
        self.perform_homography()
        self.create_meshgrid()

if __name__ == "__main__":
    ####### Pass 2 ########
    # project_name = 'data4'
    # prediction_folder = 'pass3'
    # first_calib_lane = (59, 191)
    # second_calib_lane = (17, 172)
    ##### multip points #############
    # first_calib_lane = (3, 87)
    # second_calib_lane = (16, 83)

    ######### Pass 3 ##########
    project_name = 'p3'
    prediction_folder = 'pass3'
    first_calib_lane = (212, 283)
    second_calib_lane = (161, 233)

    model = Homography(project_name, prediction_folder, first_calib_lane, second_calib_lane)
    model.run()
