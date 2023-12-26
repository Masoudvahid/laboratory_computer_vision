import matplotlib.pyplot as plt


def parse_coordinates(line):
    time_lat_lon = line[0].split(',')
    latitude = float(time_lat_lon[0][1:])
    longitude = float(time_lat_lon[1])
    return latitude, longitude


def read_gps_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip().split() for line in file.readlines()]
        return [parse_coordinates(line) for line in lines]


def plot_trajectory(coordinates, label, linestyle='-', linewidth=1.0, alpha=1.0):
    latitudes, longitudes = zip(*coordinates)
    # plt.plot(longitudes, latitudes)
    plt.scatter(longitudes, latitudes, label=label, linestyle='-', linewidth=linewidth, alpha=alpha)


lane_paths_original = ['/home/user/Documents/accurate_speed_calculation/gps/data1/calibration/lane2.csv',
                       '/home/user/Documents/accurate_speed_calculation/gps/data1/calibration/lane3.csv']
labels_original = ['lane1', 'lane2']

lane_paths_modified = ['/home/user/Documents/accurate_speed_calculation/gps/data1/calibration/lane2_modif.csv',
                       '/home/user/Documents/accurate_speed_calculation/gps/data1/calibration/lane3_modif.csv']
labels_modified = ['lane1 modified', 'lane2 modified']

plt.figure(figsize=(10, 6))

# Plot original lanes with dashed lines, half size, and half transparency
for lane_path, label in zip(lane_paths_original, labels_original):
    coordinates = read_gps_coordinates(lane_path)
    plot_trajectory(coordinates, label, linestyle=(0, (1, 1)), linewidth=0.1, alpha=0.4)

# Plot modified lanes with solid lines
for lane_path, label in zip(lane_paths_modified, labels_modified):
    coordinates = read_gps_coordinates(lane_path)
    plot_trajectory(coordinates, label)

plt.title('GPS Coordinates Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
