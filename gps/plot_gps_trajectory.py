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


def plot_trajectory(coordinates, label):
    latitudes, longitudes = zip(*coordinates)
    plt.plot(longitudes, latitudes, label=label, marker='o', markersize=0.1)
    plt.scatter(longitudes, latitudes)


# lane_paths = ['/home/user/Documents/accurate_speed_calculation/gps/data4_back/calibration/pass1.csv', '/home/user/Documents/accurate_speed_calculation/gps/data4_back/calibration/pass3.csv']
# labels = ['Track 2', 'p3']
# lane_paths = ['/home/user/Documents/accurate_speed_calculation/gps/data4_back/calibration/pass3.csv']
# labels = ['p3']

lane_paths = ['/home/user/Documents/accurate_speed_calculation/gps/data1/calibration/lane2.csv', '/home/user/Documents/accurate_speed_calculation/gps/data1/calibration/lane3.csv']
labels = ['lane1', 'lane2']



plt.figure(figsize=(10, 6))

for lane_path, label in zip(lane_paths, labels):
    coordinates = read_gps_coordinates(lane_path)
    plot_trajectory(coordinates, label)

plt.title('GPS Coordinates Trajectory')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()
