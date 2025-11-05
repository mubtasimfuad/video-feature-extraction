import cv2
import json
import os
from datetime import datetime
import numpy as np
import pytesseract
from PIL import Image


def get_video_info(video_path):
    """Extract basic video information using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return {
        "filename": os.path.basename(video_path),
        "filepath": video_path,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "duration_seconds": round(duration, 2),
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def detect_shot_cuts(video_path, threshold=0.1, sample_rate=1):
    """
    Detect hard cuts in a video.
    video_path: Path to the video file
    threshold: Difference threshold to consider as a cut (0-1)
    sample_rate: Process every Nth frame

    """
    print(f"  Detecting shot cuts (threshold={threshold})...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    cuts = []
    prev_frame = None
    frame_number = 0
    processed_frames = 0

    while True:
        """
        Read the next frame from the video
        Convert to grayscale
        Compare with previous frame using histogram comparison
        If difference exceeds threshold, record as a cut

        """
        ret, frame = cap.read()

        if not ret:
            break

        # Sample frames
        if frame_number % sample_rate != 0:
            frame_number += 1
            continue

        # Convert frame to grayscale for comparison
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            hist_prev = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
            hist_curr = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])

            # Normalize histograms
            hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()
            hist_curr = cv2.normalize(hist_curr, hist_curr).flatten()

            # correlation: 1 = identical, 0 = completely different
            correlation = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

            # If correlation is low, it's a cut
            difference = 1 - correlation

            if difference > threshold:
                cuts.append(
                    {
                        "frame": frame_number,
                        "time_seconds": round(
                            frame_number / cap.get(cv2.CAP_PROP_FPS), 2
                        ),
                        "difference": round(difference, 3),
                    }
                )

        prev_frame = gray_frame.copy()
        frame_number += 1
        processed_frames += 1

    cap.release()

    result = {
        "total_cuts": len(cuts),
        "cuts": cuts,
        "processed_frames": processed_frames,
        "threshold_used": threshold,
    }

    print(f" Found {len(cuts)} shot cuts")

    return result


def analyze_motion(video_path, sample_rate=5):
    """
    Analyze motion in video using Optical Flow.
    video_path: Path to the video file
    sample_rate: Process every Nth frame for speed
    """

    print(f"  Analyzing motion...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    motion_magnitudes = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Sample frames for speed
        if frame_number % sample_rate != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Calculate magnitude of motion
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_magnitude = np.mean(magnitude)
        motion_magnitudes.append(float(avg_magnitude))

        prev_gray = gray.copy()

    cap.release()

    # Calculate statistics
    if motion_magnitudes:
        avg_motion = float(np.mean(motion_magnitudes))
        max_motion = float(np.max(motion_magnitudes))
        min_motion = float(np.min(motion_magnitudes))
    else:
        avg_motion = max_motion = min_motion = 0.0

    print(f"\nAverage motion: {avg_motion:.4f}")

    return {
        "average_motion": round(avg_motion, 4),
        "max_motion": round(max_motion, 4),
        "min_motion": round(min_motion, 4),
        "frames_analyzed": len(motion_magnitudes),
    }


def detect_text(video_path, sample_rate=None):
    """
    Detect text in video using OCR with adaptive sampling.
    video_path: Path to the video file
    sample_rate: Process every Nth frame (None = auto-calculate)
    """

    print(f"  Detecting text...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    # Auto-calculate sample rate based on video duration
    if sample_rate is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Adaptive: Higher sample rate (less frequent) for better quality
        if duration < 10:
            sample_rate = 15  # Every 15 frames for short videos
        elif duration < 30:
            sample_rate = 20  # Every 20 frames for medium videos
        else:
            sample_rate = 30  # Every 30 frames for long videos

    # Get video resolution for adaptive thresholds
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_low_res = (width * height) < 500000
    
    min_confidence = 30 if is_low_res else 40
    min_words = 1 if is_low_res else 2

    frames_with_text = 0
    total_frames_checked = 0
    all_text_found = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % sample_rate != 0:
            continue

        total_frames_checked += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        frame_words = []
        for i, word in enumerate(ocr_data['text']):
            if word.strip():
                conf = int(ocr_data['conf'][i])
                word_clean = word.strip()
                
                if (len(word_clean) >= 3 and
                    conf > min_confidence and
                    word_clean.replace('#', '').replace('-', '').isalpha()):
                    frame_words.append(word_clean)
        
        if len(frame_words) >= min_words:
            frames_with_text += 1
            all_text_found.extend(frame_words)

    cap.release()

    text_ratio = frames_with_text / total_frames_checked if total_frames_checked > 0 else 0
    unique_words = list(set(all_text_found))

    print(f"Text present in {frames_with_text}/{total_frames_checked} frames ({text_ratio:.2%})")

    return {
        "text_present_ratio": round(text_ratio, 4),
        "frames_with_text": frames_with_text,
        "frames_checked": total_frames_checked,
        "unique_words_found": sorted(unique_words)[:20],
        "total_unique_words": len(unique_words),
        "sample_rate_used": sample_rate
    }


def main():
    video_files = [
        "videos/sample_video_for_motion.mp4",
        "videos/sample_video_for_scene_cut.mp4",
        "videos/sample_video_for_text.mp4",
    ]
    print("(Video Feature Extraction)\n")
    for video_path in video_files:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        print(f"\nProcessing video: {video_path}")
        try:
            # Get basic video info
            info = get_video_info(video_path)
            print(
                f"{info['filename']}: {info['resolution']}, {info['fps']} FPS, {info['duration_seconds']}s, {info['frame_count']} frames"
            )
            # Detect shot cuts
            cuts_result = detect_shot_cuts(video_path, threshold=0.1, sample_rate=1)
            # Analyze motion
            motion_result = analyze_motion(video_path, sample_rate=5)

            # Detect text
            text_result = detect_text(video_path, sample_rate=5)

            # Combine all features
            features = {
                "video_info": info,
                "shot_cuts": cuts_result,
                "motion_analysis": motion_result,
                "text_detection": text_result,
            }

            # Save to JSON
            os.makedirs("output", exist_ok=True)
            output_filename = (
                f"output/{os.path.splitext(info['filename'])[0]}_features.json"
            )

            with open(output_filename, "w") as f:
                json.dump(features, f, indent=4)

            print(f"Saved features to: {output_filename}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    print("Feature extraction complete!")


if __name__ == "__main__":
    main()
