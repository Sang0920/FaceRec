# Face Recognition Attendance System

Real-time face recognition system for employee attendance tracking and face recognition.

## Prerequisites

- Python 3.8+
- OpenCV
- CUDA-capable GPU (recommended)
- RTSP camera stream
- Linux system with crontab
- 4GB+ RAM

## Installation

1. Clone the repository and create virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```bash
RTSP_URL=rtsp://your-camera-url
SHIFT_NAME=your-shift-name
```

## Usage

### Manual Run

Run the script with specified duration and check type:
```bash
./run_main.sh -d <duration_seconds> -t <IN|OUT>
```

Example:
```bash
./run_main.sh -d 60 -t IN  # Run check-in for 60 seconds
```

### Automatic Scheduling

```bash
./run.sh
```
This script will run the check-in and check-out process every day at your shift time.

## Logs

Log files are stored in `logs/` directory:
- 

check-in.log

: Check-in process logs  
- 

check-out.log

: Check-out process logs

## Notes

- System automatically skips processing on holidays
- Face recognition threshold: 0.19
- Minimum profiles per track: 3
```