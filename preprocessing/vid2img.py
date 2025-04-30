import os
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_frames(video_path, output_dir):
    """
    Extract frames from a video and save them as images
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save the extracted frames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(str(video_path))
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Read frames until video is completed
    frame_count = 0
    while True:
        # Read next frame
        success, frame = video.read()
        if not success:
            break
        
        # Save frame as an image file
        output_path = os.path.join(output_dir, f"{frame_count:05d}.jpg")
        cv2.imwrite(output_path, frame)
        frame_count += 1
    
    # Release the video capture object
    video.release()
    print(f"Extracted {frame_count} frames from {video_path.name}")

def process_videos(input_dir, output_base_dir):
    """
    Process all videos in the input directory
    
    Args:
        input_dir: Directory containing video files
        output_base_dir: Base directory to save extracted frames
    """
    input_path = Path(input_dir)
    output_path = Path(output_base_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Process each video file
    video_files = list(input_path.glob("*.mp4"))
    for video_file in tqdm(video_files, desc=f"Processing videos in {input_dir}"):
        # Get video name without extension
        video_name = video_file.stem
        
        # Create output directory for this video
        video_output_dir = output_path / video_name
        
        # Extract frames from the video
        extract_frames(video_file, video_output_dir)

def main():
    # Define paths
    data_dir = Path("data")
    
    # Input directories
    train_500_dir = data_dir / "train_trimmed_500"
    train_1000_dir = data_dir / "train_trimmed_1000"
    train_1500_dir = data_dir / "train_trimmed_1500"
    test_dir = data_dir / "test_trimmed"
    
    # Output directories
    output_dir = data_dir / "images"
    train_500_output = output_dir / "train_500"
    train_1000_output = output_dir / "train_1000"
    train_1500_output = output_dir / "train_1500"
    test_output = output_dir / "test"
     
    # Create base output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process videos in each directory
    if train_500_dir.exists():
        process_videos(train_500_dir, train_500_output)
    else:
        print(f"Directory {train_500_dir} does not exist")
    
    if train_1000_dir.exists():
        process_videos(train_1000_dir, train_1000_output)
    else:
        print(f"Directory {train_1000_dir} does not exist")
    
    if train_1500_dir.exists():
        process_videos(train_1500_dir, train_1500_output)
    else:
        print(f"Directory {train_1500_dir} does not exist")
    
    if test_dir.exists():
        process_videos(test_dir, test_output)
    else:
        print(f"Directory {test_dir} does not exist")

if __name__ == "__main__":
    main()
