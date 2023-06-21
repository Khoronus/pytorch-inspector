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

def combine_videos(video_files, output_file, output_size, out_video):
    """
    Args:
    - **video_files**: List of videos to open.
    - **output_file**: Output video filename.
    - **output_size**: Output video size.
    - **out_video**: cv2.VideoWriter expected if new frames should be added.
    It combines all valid videos passed with video_files in a single video.
    The output video is created only if out_video is None.

    Return
    - It returns the updated video output.
    """    
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
    if out_video is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_file, fourcc, 20.0, output_size)
    
    while True:
        # Create an empty frame for the output video
        output_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
        # Read a frame from each video and add it to the output frame
        for i, video in enumerate(videos):
            # unable to open the video
            if not video.isOpened():
                print('video not opened')
                continue
            print(f'video id:{i}')
            ret, frame = video.read()
            if not ret:
                print('video not ret')
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
        out_video.write(output_frame)
        
        # Break the loop if any of the videos has ended
        if not ret:
            break
    
    # Release everything if job is finished
    for video in videos:
        video.release()

    return out_video
    

def main():
    print(f'{os.path.basename(__file__)} [folder_in] [filename_out] [width] [height] [indexes_desired(0,1,...)] [headers_desired] [headers_excluded]')
    print(f'i.e. output output_file.mp4 1920 1080 -1 -1 -1')
    arg1 = sys.argv[1]
    print("Argument 1:", arg1)
    arg2 = sys.argv[2]
    print("Argument 2:", arg2)
    arg3 = sys.argv[3]
    print("Argument 3:", arg3)
    arg4 = sys.argv[4]
    print("Argument 4:", arg4)
    arg5 = sys.argv[5]
    print("Argument 5:", arg5)
    arg6 = sys.argv[6]
    print("Argument 6:", arg6)
    arg7 = sys.argv[7]
    print("Argument 7:", arg7)

    # Get the list of the files
    files = get_files_in_folder(arg1)

    # Get the indexes selected
    indexes_desired = [int(x) for x in arg5.split(',')]
    print(f'indexes_desired:{indexes_desired}')
    # Get the header selected
    header_desired = [x for x in arg6.split(',')]
    print(f'header_desired:{header_desired}')
    # Get the header excluded
    header_excluded = [x for x in arg7.split(',')]
    print(f'header_excluded:{header_excluded}')

    # Create a video with all the selected indexes
    out_video = None
    for index_desired in indexes_desired:
        
        # Use all the files or prune the videos which index is different than the desired one
        if index_desired < 0:
            valid_files = files
        else:
            valid_files = []
            for file in files:
                
                try:
                    filename = os.path.basename(file)
                    print(f'filename:{filename}')
                    separator = '_'
                    words = filename.split(separator)
                    # Get the index of the video
                    # For more information, please check the DataRecorder.py code
                    # where the file is created (fname_out = ...)
                    if index_desired < 0 or int(words[-2]) == index_desired:
                        if header_desired[0] == "-1" or words[-5] in header_desired:
                            if header_excluded[0] == "-1" or words[-5] not in header_excluded:
                                valid_files.append(file)
                except Exception as e:
                    print(f'exception:{e}')
        print(f'index_desired:{index_desired} valid_files:{valid_files}')
        if len(valid_files) > 0:
            sorted_valid_files = sorted(valid_files)
            out_video = combine_videos(sorted_valid_files, arg2, (int(arg3),int(arg4)), out_video)
    # Release the video
    if out_video is not None:
        out_video.release()

if __name__ == '__main__':
    main()