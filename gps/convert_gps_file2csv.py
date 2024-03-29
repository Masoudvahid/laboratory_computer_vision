import csv

input_file_path = f'/home/user/Documents/accurate_speed_calculation/gps/data1/lane1.txt'
output_csv_path = f'/home/user/Documents/accurate_speed_calculation/gps/data1/lane1.csv'

with open(input_file_path, 'r') as input_file, open(output_csv_path, 'w', newline='') as output_csv:
    csv_writer = csv.writer(output_csv)
    # csv_writer.writerow(['time', 'lat', 'long'])

    for line in input_file:
        data = line.split()
        time, lat, long = data[1], data[2], data[3]
        csv_writer.writerow([time, lat, long])
