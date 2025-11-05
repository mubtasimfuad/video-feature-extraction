# Video Feature Extraction Tool

A Python tool that analyzes video files and extracts visual and temporal features like scene changes, motion, and text.

## Features

**1. Shot Cut Detection**
- Detects scene changes (hard cuts) in videos
- Uses histogram comparison between frames
- Returns cut locations with timestamps

**2. Motion Analysis**
- Quantifies movement using Optical Flow
- Calculates average motion magnitude
- Useful for activity detection

**3. Text Detection (OCR)**
- Extracts text from video frames
- Uses Tesseract OCR engine
- Returns text presence ratio and detected words

## Requirements

- Python 3.11+
- OpenCV
- NumPy
- pytesseract
- Tesseract OCR (system dependency)

## Installation

### 1. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Verify installation
tesseract --version
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python extract_features.py
```

The script will:
1. Process all videos in the `videos/` folder
2. Extract features from each video
3. Save results as JSON files in `output/` folder

### Output Format

Each video generates a JSON file with:

```json
{
    "video_info": {
        "filename": "example.mp4",
        "resolution": "1920x1080",
        "fps": 30.0,
        "duration_seconds": 10.5
    },
    "shot_cuts": {
        "total_cuts": 3,
        "cuts": [...]
    },
    "motion_analysis": {
        "average_motion": 5.23
    },
    "text_detection": {
        "text_present_ratio": 0.25,
        "unique_words_found": [...]
    }
}
```

## Project Structure

```
video-feature-extraction/
├── videos/              # Place your video files here
├── output/              # JSON output files (auto-generated)
├── extract_features.py  # Main script
├── requirements.txt    # Python dependencies
└── README.md
```

## Notes

- **Video formats**: Supports MP4, AVI, MOV (any OpenCV-compatible format)
- **Processing time**: Depends on video length and resolution
- **Text detection**: Accuracy varies with video quality and text clarity
- **Sample videos**: Add your videos to the `videos/` folder

## Limitations

- Text detection works best with clear, high-contrast text
- Low-resolution videos may have reduced OCR accuracy
- Very long videos may take time to process

## Author

Mubtasim Fuad

## License

MIT
