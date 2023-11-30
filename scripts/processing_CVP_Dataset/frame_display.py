import os
import cv2
import shutil

# ---- Utility Functions ----

def get_files_in_folder(source_folder, filenames):
    """Returns the paths of the specified filenames present in the given folder."""
    return [os.path.join(source_folder, name) for name in filenames if os.path.exists(os.path.join(source_folder, name))]


def create_output_folder(target_base_folder, folder_name):
    """Creates and returns the path of the new output folder."""
    new_folder = os.path.join(target_base_folder, folder_name)
    os.makedirs(new_folder, exist_ok=True)
    return new_folder


def process_folders(source_folder, target_base_folder):
    """Processes folders to identify videos to be processed and create corresponding output folders."""
    all_input_files = []
    all_output_files = []

    for folder in os.listdir(source_folder):
        current_folder = os.path.join(source_folder, folder)

        # Skip if it's not a directory
        if not os.path.isdir(current_folder):
            continue

        print(f"Processing folder: {current_folder}")

        new_folder = create_output_folder(target_base_folder, folder)

        # Copy data.csv if exists
        csv_path = os.path.join(current_folder, 'data.csv')
        if os.path.exists(csv_path):
            shutil.copy2(csv_path, new_folder)

        # Identify video files and their output paths
        input_files = get_files_in_folder(current_folder, ['K1.mkv', 'K2.mkv'])
        output_files = [os.path.join(new_folder, os.path.basename(f).replace('.mkv', '_Cropped_Colour.mkv')) for f in input_files]

        all_input_files.extend(input_files)
        all_output_files.extend(output_files)

    return all_input_files, all_output_files


def display_first_frame(filename, rect_size, output_filename):
    """Displays the first frame of a video and lets the user select a cropping region."""
    video_stream = cv2.VideoCapture(filename)

    # Extract the tenth frame
    for _ in range(10):
        ret, frame = video_stream.read()

    if not ret:
        print("Error reading the video file.")
        return

    original_frame_unscaled = frame.copy()
    scale = 0.5
    scaled_frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    param = {
        "original_frame_unscaled": original_frame_unscaled,
        "frame_with_rectangle": scaled_frame,
        "rect_size": rect_size,
        "input_filename": filename,
        "output_filename": output_filename,
        "scale": scale
    }

    cv2.namedWindow('First Color Frame')
    cv2.setMouseCallback('First Color Frame', on_mouse, param)
    cv2.imshow('First Color Frame', scaled_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    video_stream.release()

    return param.get("click_coordinates")


def on_mouse(event, x, y, flags, param):
    """Mouse callback function to draw cropping rectangle and capture click coordinates."""
    scale, rect_size = param["scale"], param["rect_size"]
    x_unscaled, y_unscaled = int(x / scale), int(y / scale)
    
    if event == cv2.EVENT_MOUSEMOVE:
        x_start, y_start = x_unscaled - rect_size[0] // 2, y_unscaled - rect_size[1] // 2

        param["frame_with_rectangle"] = param["original_frame_unscaled"].copy()
        cv2.rectangle(param["frame_with_rectangle"], (x_start, y_start),
                      (x_start + rect_size[0], y_start + rect_size[1]), (0, 255, 0), 2)
        
        display_frame = cv2.resize(param["frame_with_rectangle"], (int(param["frame_with_rectangle"].shape[1] * scale), int(param["frame_with_rectangle"].shape[0] * scale)))
        cv2.imshow('First Color Frame', display_frame)

    elif event == cv2.EVENT_LBUTTONDOWN:
        param["click_coordinates"] = (x_unscaled - rect_size[0] // 2, y_unscaled - rect_size[1] // 2)
        cv2.destroyAllWindows()
