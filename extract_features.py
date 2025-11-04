import cv2
import json
import os
from datetime import datetime

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
        'filename': os.path.basename(video_path),
        'filepath': video_path,
        'fps': round(fps, 2),
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}",
        'duration_seconds': round(duration, 2),
        'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def main():
    video_files = [
        'videos/sample_video_for_motion.mp4',
        'videos/sample_video_for_scene_cut.mp4',
        'videos/sample_video_for_text.mp4'
    ]
    for video_path in video_files:
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        try:
            info = get_video_info(video_path)
            print(f"{info['filename']}: {info['resolution']}, {info['fps']} FPS, {info['duration_seconds']}s, {info['frame_count']} frames")
            os.makedirs('output', exist_ok=True)
            output_filename = f"output/{os.path.splitext(info['filename'])[0]}_info.json"
            with open(output_filename, 'w') as f:
                json.dump(info, f, indent=4)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
