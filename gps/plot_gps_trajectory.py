import matplotlib.pyplot as plt


def parse_coordinates(line):
    time_lat_lon = line[0].split(',')
    latitude = float(time_lat_lon[1][1:])
    longitude = float(time_lat_lon[2])
    return latitude, longitude


def read_gps_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip().split() for line in file.readlines()]
        return [parse_coordinates(line) for line in lines]


def plot_trajectory(coordinates, label):
    latitudes, longitudes = zip(*coordinates)
    plt.plot(longitudes, latitudes, label=label)
    plt.scatter(longitudes, latitudes)


lane_paths = ['lane2.csv', 'lane3.csv']
labels = ['Track 2', 'Track 3']

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
