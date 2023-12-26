import re

def extract_coords(line):
    # Regular expression to match GPRMC lines and extract coordinates
    pattern = r'^\$GPRMC,(\d+\.\d+),A,(\d+\.\d+),([NS]),(\d+\.\d+),([EW]),.*$'

    match = re.match(pattern, line)
    if match:
        time, lat, lat_dir, lon, lon_dir = match.groups()

        # Check if lat_dir is 'S', if so, add a negative sign
        lat_sign = '' if lat_dir == 'N' else '-'

        # Check if lon_dir is 'W', if so, add a negative sign
        lon_sign = '' if lon_dir == 'E' else '-'

        return f"{time} {lat_sign}{lat} {lon_sign}{lon}"
    else:
        return None

def parse_file(file_path, output_file):
    with open(file_path, 'r') as file, open(output_file, 'w') as output:
        for line in file:
            if line.startswith('$GPRMC'):
                coords = extract_coords(line)
                if coords:
                    output.write(coords + '\n')

if __name__ == "__main__":
    file_path = "/home/user/Documents/accurate_speed_calculation/gps/data4/gps_white.txt"
    output_file = "/home/user/Documents/accurate_speed_calculation/gps/data4/gps_white_out.txt"
    parse_file(file_path, output_file)
