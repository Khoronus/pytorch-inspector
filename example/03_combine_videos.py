import os
import sys
import numpy as np
import cv2

def get_files_in_folder(folder_path):
    files = []
    for entry in os.scandir(folder_path):
        if entry.is_file():
            files.append(entry.path)
    return files

def combine_videos(video_files, output_file, output_size):
    # Load the videos
    videos = [cv2.VideoCapture(video_file) for video_file in video_files]
    
    # Get the dimensions of the first frame of the first open video
    for video in videos:
        if not video.isOpened():
            continue
        ret, frame = video.read()
        height, width, _ = frame.shape
        break
    
    # Calculate the size of each panel in the output video
    n_videos = len(videos)
    n_rows = int(np.ceil(np.sqrt(n_videos)))
    n_cols = int(np.ceil(n_videos / n_rows))
    panel_width = output_size[0] // n_cols
    panel_height = output_size[1] // n_rows
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, output_size)
    
    while True:
        # Create an empty frame for the output video
        output_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
        # Read a frame from each video and add it to the output frame
        for i, video in enumerate(videos):
            # unable to open the video
            if not video.isOpened():
                continue
            print(f'video id:{i}')
            ret, frame = video.read()
            if not ret:
                break
            # Resize the frame to fit in the panel
            frame = cv2.resize(frame, (panel_width, panel_height))
            row = i // n_cols
            col = i % n_cols
            x_start = col * panel_width
            y_start = row * panel_height
            x_end = x_start + panel_width
            y_end = y_start + panel_height
            output_frame[y_start:y_end, x_start:x_end] = frame
        
        # Write the output frame to the output video
        out.write(output_frame)
        
        # Break the loop if any of the videos has ended
        if not ret:
            break
    
    # Release everything if job is finished
    for video in videos:
        video.release()
    out.release()

def main():
    print(f'{os.path.basename(__file__)} [folder_in] [filename_out] [width] [height]')
    print(f'i.e. output output_file.mp4 1920 1080')
    arg1 = sys.argv[1]
    print("Argument 1:", arg1)
    arg2 = sys.argv[2]
    print("Argument 2:", arg2)
    arg3 = sys.argv[3]
    print("Argument 3:", arg3)
    arg4 = sys.argv[4]
    print("Argument 4:", arg4)
    files = get_files_in_folder(arg1)
    print(f'files:{files}')
    combine_videos(files, arg2, (int(arg3),int(arg4)))

if __name__ == '__main__':
    main()