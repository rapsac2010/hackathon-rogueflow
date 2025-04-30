import math  # Import math for isnan
import os
import subprocess  # Added import

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm  # Optional: for progress bar


# Helper function to sample N frames uniformly from a video
def sample_uniform_frames(video_path, num_frames=15):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return frames  # edge case: no frames
    # Compute frame indices to sample
    indices = np.linspace(0, frame_count-1, num=num_frames)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to 224x224 and convert BGR to RGB for keras preprocess
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)
    
def get_video_duration_opencv(filepath):
    """Gets the duration of a video file using OpenCV."""
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            print(f"Error: Could not open video file {filepath}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)      # Frames per second
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps is None or fps == 0 or frame_count is None:
                print(f"Warning: Could not get FPS or frame count for {filepath}")
                # Fallback or estimate if needed, otherwise return None
                cap.release()
                return None # Or handle appropriately

        duration = frame_count / fps  # Duration in seconds
        cap.release()
        return duration
    except Exception as e:
        print(f"Error processing file {filepath} with OpenCV: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return None

        # # Example usage:
        # video_path = 'path/to/your/video.mp4'
        # duration_seconds = get_video_duration_opencv(video_path)

        # if duration_seconds is not None:
        #     print(f"The video duration is: {duration_seconds} seconds")


def trim_videos(train_dir, output_dir, df_full, num_frames_before, cutoff_time_column):
    """
    Trims training videos to a fixed number of frames based on a cutoff time.
    The output video will contain exactly `num_frames_before + 1` frames,
    unless the original video is shorter.

    If cutoff_time_random is NaN, the last `num_frames_before + 1` frames are taken.
    If cutoff_time_random is valid, the clip ends at the cutoff frame, but the
    window is shifted if necessary to maintain the target frame count.

    Args:
        train_dir (str): Path to the directory containing the original training videos.
        output_dir (str): Path to the directory where trimmed videos will be saved.
        df_full (pd.DataFrame): DataFrame containing video filenames and cutoff times.
                                 Expected columns: 'id', 'cutoff_time_random'.
        num_frames_before (int): Number of frames to include *before* the cutoff frame.
                                 The total output frames will be num_frames_before + 1.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        cutoff_map = pd.Series(df_full[cutoff_time_column].values, index=df_full['file']).to_dict()
    except KeyError:
        print("Error: DataFrame 'df_full' must contain 'id' and 'cutoff_time_random' columns.")
        return

    print(f"Processing {len(cutoff_map)} videos listed in the DataFrame...")

    processed_count = 0
    not_found_count = 0
    error_count = 0
    target_frames = num_frames_before + 1 # Define the target frame count N

    for filename_id, cutoff_time in tqdm(cutoff_map.items()):
        filename = str(filename_id) # Ensure filename is string
        input_path = os.path.join(train_dir, filename)

        if not os.path.exists(input_path):
            not_found_count += 1
            continue

        output_path = os.path.join(output_dir, filename)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {filename}")
            error_count += 1
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"Warning: Could not read FPS for {filename}. Assuming 30 FPS.")
            fps = 30

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"Error: Video {filename} reported 0 or negative frames ({total_frames}). Skipping.")
            error_count += 1
            cap.release()
            continue

        start_frame = 0
        end_frame = 0 # Use end_frame instead of cutoff_frame for clarity in this context
        num_frames_to_write = 0

        # --- Calculate Start and End Frames ---
        if total_frames < target_frames:
            print(f"Warning: Video {filename} has only {total_frames} frames, less than the target {target_frames}. Taking all frames.")
            start_frame = 0
            end_frame = total_frames - 1
        else:
            # Video is long enough, determine window based on cutoff_time
            is_nan_cutoff = isinstance(cutoff_time, float) and math.isnan(cutoff_time)

            if is_nan_cutoff:
                # Handle NaN: Take the last target_frames frames
                # print(f"Info: NaN cutoff for {filename}. Taking last {target_frames} frames.")
                end_frame = total_frames - 1
                start_frame = total_frames - target_frames
            else:
                # Handle valid cutoff time: Center the window around the cutoff if possible
                cutoff_frame_ideal = int(cutoff_time * fps)
                start_frame_ideal = cutoff_frame_ideal - num_frames_before # = cutoff_frame_ideal - (target_frames - 1)

                if start_frame_ideal < 0:
                    # Window hits the start, take first target_frames
                    # print(f"Info: Cutoff time for {filename} is early. Taking first {target_frames} frames.")
                    start_frame = 0
                    end_frame = target_frames - 1
                elif cutoff_frame_ideal >= total_frames:
                    # Window hits or exceeds the end, take last target_frames
                    # print(f"Info: Cutoff time for {filename} is late/beyond end. Taking last {target_frames} frames.")
                    end_frame = total_frames - 1
                    start_frame = total_frames - target_frames
                else:
                    # Ideal window is within bounds
                    start_frame = start_frame_ideal
                    end_frame = cutoff_frame_ideal

        # Final boundary check (redundant but safe)
        start_frame = max(0, start_frame)
        end_frame = min(end_frame, total_frames - 1)
        num_frames_to_write = end_frame - start_frame + 1

        # If, after all calculations, we somehow don't have the target number of frames
        # (only possible if original video was shorter), handle it.
        if num_frames_to_write <= 0:
             print(f"Error: Calculated 0 or negative frames to write for {filename} (start={start_frame}, end={end_frame}). Skipping.")
             error_count += 1
             cap.release()
             continue

        # --- Set up Writer and Read/Write Frames ---
        # Set the starting position accurately
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # Verify position (optional, but useful for debugging)
        # current_pos_check = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # if abs(current_pos_check - start_frame) > 1: # Allow for slight inaccuracy
        #     print(f"Warning: Could not accurately seek to start frame {start_frame} for {filename}. Current position: {current_pos_check}")


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"Error: Could not open video writer for: {output_path}")
            error_count += 1
            cap.release()
            continue

        success = True
        frames_written = 0
        for i in range(num_frames_to_write):
            ret, frame = cap.read()
            if not ret:
                # This indicates an issue reading the frame (e.g., end of stream earlier than expected, corrupt frame)
                print(f"Warning: Failed to read frame at index {start_frame + i} (frame {i+1} of {num_frames_to_write}) from {filename}. Total frames reported: {total_frames}. Stopping trim.")
                success = (frames_written > 0) # Consider successful only if some frames were written
                break
            writer.write(frame)
            frames_written += 1

        cap.release()
        writer.release()

        # --- Update Summary ---
        if success and frames_written > 0:
             if frames_written < num_frames_to_write:
                  print(f"Warning: Wrote only {frames_written} frames, but expected {num_frames_to_write} for {filename}. Original video might be shorter than reported or read error occurred.")
                  # This might indicate an issue, potentially remove the file
                  if os.path.exists(output_path):
                      print(f"Removing incomplete file due to read error: {output_path}")
                      os.remove(output_path)
                      error_count += 1 # Count as error if incomplete
                  else:
                     processed_count += 1 # Or count as processed if file removal isn't desired
             else:
                  # Successfully wrote the expected number of frames
                  processed_count += 1
        else:
             # Either success was False or 0 frames were written
             error_count += 1
             if os.path.exists(output_path):
                  print(f"Removing potentially empty or failed file: {output_path}")
                  os.remove(output_path)


    print("\n--- Processing Summary ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Files not found:       {not_found_count}")
    print(f"Errors during processing: {error_count}")
    print("------------------------")



def trim_videos_ffmpeg(train_dir, output_dir, df_full, num_frames_before, cutoff_time_column):
    """
    Trims training videos efficiently using ffmpeg to approximately 6 seconds,
    ending near the specified cutoff time. Uses stream copying for speed.

    If cutoff_time is NaN, the last ~6 seconds are taken.
    If cutoff_time is valid, the clip aims to end at cutoff_time, spanning
    6 seconds prior. Adjustments are made if this window hits video boundaries.

    Args:
        train_dir (str): Path to the directory containing the original training videos.
        output_dir (str): Path to the directory where trimmed videos will be saved.
        df_full (pd.DataFrame): DataFrame containing video filenames and cutoff times.
                                 Expected columns: 'file', cutoff_time_column.
        num_frames_before (int): *Not used in this time-based version, but kept for signature consistency.*
        cutoff_time_column (str): Name of the column containing the cutoff time.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        if 'file' not in df_full.columns:
             raise KeyError("DataFrame 'df_full' must contain a 'file' column.")
        if cutoff_time_column not in df_full.columns:
             raise KeyError(f"DataFrame 'df_full' must contain the specified cutoff time column: '{cutoff_time_column}'.")
        cutoff_map = pd.Series(df_full[cutoff_time_column].values, index=df_full['file']).to_dict()
    except KeyError as e:
        print(f"Error: {e}")
        return

    print(f"Processing {len(cutoff_map)} videos listed in the DataFrame using ffmpeg (time-based ~6s trim)...")

    processed_count = 0
    not_found_count = 0
    error_count = 0
    target_duration_sec = 6.0 # Desired output duration

    for filename_id, cutoff_time in tqdm(cutoff_map.items()):
        filename = str(filename_id)
        input_path = os.path.join(train_dir, filename)

        if not os.path.exists(input_path):
            not_found_count += 1
            continue

        output_path = os.path.join(output_dir, filename)

        # --- Get video duration (more robustly than just frames/fps) ---
        total_duration_sec = None
        try:
            # Use ffprobe to get accurate duration
            ffprobe_command = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                input_path
            ]
            result = subprocess.run(ffprobe_command, check=True, capture_output=True, text=True)
            total_duration_sec = float(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            # Fallback to OpenCV if ffprobe fails
            print(f"Warning: ffprobe failed for {filename} ({e}). Falling back to OpenCV duration.")
            total_duration_sec = get_video_duration_opencv(input_path)

        if total_duration_sec is None or total_duration_sec <= 0:
            print(f"Error: Could not determine valid duration for {filename}. Skipping.")
            error_count += 1
            continue

        # --- Calculate Start Time and Duration for ffmpeg ---
        start_time_sec = 0.0
        duration_sec = 0.0
        is_nan_cutoff = isinstance(cutoff_time, float) and math.isnan(cutoff_time)

        if is_nan_cutoff:
            # Handle NaN: Take the last target_duration_sec seconds
            start_time_sec = max(0.0, total_duration_sec - target_duration_sec)
            duration_sec = min(target_duration_sec, total_duration_sec)
            # print(f"Info: NaN cutoff for {filename}. Taking last {duration_sec:.2f}s (from {start_time_sec:.2f}s). Total: {total_duration_sec:.2f}s")
        else:
            # Handle valid cutoff time
            cutoff_time_sec = float(cutoff_time) # Ensure it's float

            # Ensure cutoff_time_sec is within video bounds for calculation
            effective_cutoff_time = min(cutoff_time_sec, total_duration_sec)

            ideal_start_time_sec = effective_cutoff_time - target_duration_sec

            if ideal_start_time_sec < 0:
                # Window hits the start of the video
                start_time_sec = 0.0
                duration_sec = min(effective_cutoff_time, target_duration_sec, total_duration_sec)
                # print(f"Info: Cutoff {cutoff_time_sec:.2f}s for {filename} is early. Taking first {duration_sec:.2f}s. Total: {total_duration_sec:.2f}s")
            else:
                # Window starts within the video
                start_time_sec = ideal_start_time_sec
                duration_sec = target_duration_sec # Aim for the full target duration
                # Adjust duration if the calculated end time (start + duration) exceeds total duration slightly due to seeking inaccuracy potential
                if start_time_sec + duration_sec > total_duration_sec:
                     duration_sec = total_duration_sec - start_time_sec
                # print(f"Info: Cutoff {cutoff_time_sec:.2f}s for {filename}. Taking {duration_sec:.2f}s (from {start_time_sec:.2f}s). Total: {total_duration_sec:.2f}s")


        # Ensure duration is positive
        duration_sec = max(0.01, duration_sec) # Avoid zero or negative duration

        # --- Execute ffmpeg command ---
        try:
            # print(f"start_time_sec: {start_time_sec}, duration_sec: {duration_sec}") # Keep for debugging if needed

            # --- Accurate Time Trimming (Re-encoding) ---
            # Place -ss AFTER -i for accurate seeking.
            # Remove -c copy to force re-encoding, ensuring time precision.
            # This is slower but more reliable for exact time cuts.
            # Try a faster preset like 'superfast' or 'ultrafast' for speed.
            command = [
                'ffmpeg',
                '-i', input_path,                 # Input first
                '-ss', str(start_time_sec),      # Then accurate seek
                '-t', str(duration_sec),         # Duration to encode
                '-c:v', 'libx264', '-crf', '23',  # Example: H.264 video, quality 23
                '-c:a', 'aac', '-b:a', '128k',    # Example: AAC audio, 128kbps bitrate
                '-preset', 'superfast',          # <--- CHANGED FROM 'fast'
                '-loglevel', 'error',            # Suppress verbose output
                '-y',                            # Overwrite output
                output_path
            ]

            result = subprocess.run(command, check=True, capture_output=True, text=True)
            processed_count += 1

        except subprocess.CalledProcessError as e:
            print(f"Error processing {filename} with ffmpeg: {e}")
            print(f"ffmpeg stderr: {e.stderr}")
            error_count += 1
            if os.path.exists(output_path):
                try: os.remove(output_path); print(f"Removed incomplete output file: {output_path}")
                except OSError as remove_err: print(f"Error removing incomplete file {output_path}: {remove_err}")
        except FileNotFoundError:
            print("Error: ffmpeg/ffprobe command not found. Make sure they are installed and in your system's PATH.")
            return
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")
            error_count += 1

    # --- Final Summary ---
    print("\n--- Processing Summary (ffmpeg - time-based ~6s trim) ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Files not found:       {not_found_count}")
    print(f"Errors during processing: {error_count}")
    print("---------------------------------------------------------")