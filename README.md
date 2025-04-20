# WhisperX Timestamped Transcripts

This tool turns **any video you drop into `./videos`** into a fully timestamped
markdown transcript written to `./transcripts/<video_name>/<video_name>.md`.

The markdown now starts with the **complete transcript text**, followed by
segment‑level sections and word‑level timestamps — perfect for searching or
auto‑clipping your long‑form content.

## Setup

1. **Clone & enter the repo**

   ```bash
   git clone https://github.com/kingbootoshi/timestamped-transcripts.git
   cd timestamped-transcripts
   ```

2. **Install FFmpeg**

   ```bash
   # macOS (Homebrew)
   brew install ffmpeg
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   # Windows (Chocolatey)
   choco install ffmpeg
   ```

3. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: .\venv\Scripts\activate
   ```

4. **Install Python deps**

   ```bash
   pip install -r requirements.txt
   ```

5. **Create the I/O folders**

   ```bash
   mkdir -p videos transcripts
   ```

## Usage

### Batch mode (recommended)

Simply drop one or more videos ( `.mp4 .mov .mkv .avi .mp3 .wav` ) into the
`videos/` folder and run:

```bash
python main.py
```

Each video produces:

```
transcripts/
└── my_video/
    └── my_video.md
```

### Single‑file mode

```bash
python main.py /path/to/video.mp4
```

Optional flags:

* `--output_file custom.md` – override the default output path  
* `--model_size tiny|base|small|medium|large`  
* `--language <lang_code>`  
* `--overwrite` – regenerate even if the transcript already exists  

## Output Format

```markdown
# Full Transcript
<entire text>

# Timestamped Transcript

## Segment 1: [00:00:00 - 00:00:12]
Segment text…

### Word‑level timestamps
- Hello: 00:00:00
- world: 00:00:01

…

## Summary
Total segments: 123
Total duration: 01:23:45
```

## Notes

* Long videos are chunked transparently; no manual splitting required.  
* A CUDA GPU is used automatically if available for faster inference.  
* Each processed video gets its own folder inside `transcripts/` to avoid clutter.