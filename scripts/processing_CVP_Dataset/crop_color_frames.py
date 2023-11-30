"""
Video Processing Script for Kinect Azure MKV files

Overview:
---------
This script processes a directory of Kinect Azure MKV video files to display the first frame of each video. The user 
can then select a cropping region by moving the mouse over the frame. Once the desired region is selected, the user 
clicks the mouse button, and the script crops the video according to the selected region. The cropped video is then 
saved in a corresponding directory structure under a target folder.

Usage:
------
1. Execute the script from the command line.
2. Provide the following arguments:
   --raw_dataset: The path to the directory containing the raw video data.
   --processed_dataset: The path to the directory where the processed (cropped) videos should be saved.
   --rect_size (optional): The size of the rectangle to be used for cropping. Default is 500x500.

Example:
--------
python script_name.py --raw_dataset path/to/raw_data --processed_dataset path/to/output_data --rect_size 500 500

After executing the script:
1. The first color frame of each video will be displayed.
2. Move the mouse to select the desired cropping region.
3. Click the mouse button to crop the video according to the selected region.
4. The script will then save the cropped video in the corresponding directory under the target folder.

Dependencies:
-------------
- OpenCV
- multiprocessing
- argparse
- shutil

Author:
-------
Coen Arrow
coen.arrow@research.uwa.edu.au

Date Created:
-------------
20/08/2023
"""

from frame_display import process_folders, display_first_frame
import argparse
from multiprocessing import Process, Semaphore, Manager, Queue
import cv2
import time
import multiprocessing

# ---- Core Functions ----

def crop_color_frames(input_filename, click_coordinates, rect_size, output_filename, progress_dict):
    """Crops video frames based on selected coordinates."""
    video_stream = cv2.VideoCapture(input_filename)
    total_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_stream = cv2.VideoWriter(output_filename, fourcc, video_stream.get(cv2.CAP_PROP_FPS), rect_size)

    x, y = click_coordinates
    w, h = rect_size
    
    while True:
        ret, frame = video_stream.read()
        if not ret:
            break

        current_frame = int(video_stream.get(cv2.CAP_PROP_POS_FRAMES))
        progress_percentage = (current_frame / total_frames) * 100
        progress_dict[input_filename] = f'{progress_percentage:.2f}% completed' # <-- Update the shared dict

        cropped_frame = frame[y:y+h, x:x+w]
        output_stream.write(cropped_frame)

    video_stream.release()
    output_stream.release()
    print(f"\nCropped video saved as {output_filename}.")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process Kinect Azure MKV video.")
    parser.add_argument('--raw_dataset', type=str, required=True, help='Path to the raw dataset directory')
    parser.add_argument('--processed_dataset', type=str, required=True, help='Path to the processed dataset directory')
    parser.add_argument('--rect_size', type=int, nargs=2, default=[500, 500], help='Rectangle size for cropping (width, height)')
    return parser.parse_args()

def crop_color_frames_with_semaphore(input_file, click_coordinates, rect_size, output_file, semaphore, progress_dict):
    try:
        crop_color_frames(input_file, click_coordinates, rect_size, output_file, progress_dict)
    finally:
        # Release the semaphore after the function finishes
        semaphore.release()

def worker(queue, semaphore, RECT_SIZE, progress_dict):
    """Worker function to process cropping tasks."""
    while True:
        data = queue.get()
        if data is None:  # sentinel to terminate the worker
            break
        input_file, click_coordinates, output_file = data

        crop_color_frames_with_semaphore(input_file, click_coordinates, RECT_SIZE, output_file, semaphore, progress_dict)

if __name__ == "__main__":
    print("Script Started")
    
    # Required for Windows
    multiprocessing.freeze_support()

    # Initialization and parsing arguments
    args = parse_arguments()
    INPUT_FILES, OUTPUT_FILES = process_folders(args.raw_dataset, args.processed_dataset)
    RECT_SIZE = tuple(args.rect_size)

    # Manager for shared objects
    manager = Manager()
    progress_dict = manager.dict()

    print(f"Input Files: {INPUT_FILES}")
    print(f"Output Files: {OUTPUT_FILES}")

    # Semaphore to limit the number of concurrent processes
    semaphore = Semaphore(3)

    # Queue for tasks
    task_queue = Queue()

    # Maintain a list of worker processes
    worker_processes = []

    # Spawn worker processes to handle cropping
    num_workers = 3  # adjust as needed
    for _ in range(num_workers):
        p = Process(target=worker, args=(task_queue, semaphore, RECT_SIZE, progress_dict))
        p.start()
        worker_processes.append(p)

    # Populate the queue with tasks
    for idx, input_file in enumerate(INPUT_FILES):
        print(f"Displaying frame for video: {input_file}")
        click_coordinates = display_first_frame(input_file, RECT_SIZE, OUTPUT_FILES[idx])

        if click_coordinates:
            task_queue.put((input_file, click_coordinates, OUTPUT_FILES[idx]))

        # Print out the progress
        time.sleep(0.1)  # Adjust this duration as needed
        for video, progress in progress_dict.items():
            print(f'{video}: {progress}', end='\r')

    # Put sentinels in the queue to terminate the worker processes
    for _ in range(num_workers):
        task_queue.put(None)

    # Join the worker processes after placing the sentinels
    for p in worker_processes:
        p.join()

    print("Script Finished")

    # As workers are now running in the background, you can proceed with other tasks 
    # or simply let the script run until all workers finish processing.