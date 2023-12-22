import cv2
import os

def create_video(frames_folder, output_video_path, fps=30):
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])  # Adjust the file extension accordingly
    if not frame_files:
        print("No frame files found in the specified folder.")
        return

    frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec based on your preference
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video created successfully: {output_video_path}")

# Example usage:
frames_folder_path = f"/home/user/Documents/accurate_speed_calculation/gps/p2/calibration/line_3"
output_video_path = f"/home/user/Documents/accurate_speed_calculation/gps/p2/calibration/line_3.mkv"
create_video(frames_folder_path, output_video_path)
for i in range(9, 11):
    pass
    frames_folder_path = f"/home/user/Documents/accurate_speed_calculation/gps/p2/calibration/line_1"
    output_video_path = f"/home/user/Documents/accurate_speed_calculation/gps/p2/calibration/line_1.mkv"
    create_video(frames_folder_path, output_video_path)
