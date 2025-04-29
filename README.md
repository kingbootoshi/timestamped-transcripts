# WhisperX Timestamped Transcripts

This tool turns **any video or audio file you drop into `./videos`** into a fully timestamped markdown transcript written to `./transcripts/<video_name>/<video_name>.md`.

It now includes **optional speaker diarization** using Pyannote.audio to identify who spoke when.

The markdown starts with the **complete transcript text** (with speaker labels if diarization is enabled), followed by segment‑level sections and word‑level timestamps — perfect for searching or auto‑clipping your long‑form content.

## Setup

1.  **Clone & enter the repo**

    ```bash
    git clone https://github.com/kingbootoshi/timestamped-transcripts.git
    cd timestamped-transcripts
    ```

2.  **Install FFmpeg**

    ```bash
    # macOS (Homebrew)
    brew install ffmpeg
    # Ubuntu/Debian
    sudo apt update && sudo apt install ffmpeg
    # Windows (Chocolatey)
    choco install ffmpeg
    ```

3.  **Create a virtual environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate      # Windows: .\venv\Scripts\activate
    ```

4.  **Install Python deps**

    ```bash
    pip install -r requirements.txt
    ```

5.  **(Optional) Hugging Face Setup for Speaker Diarization**

    If you plan to use speaker diarization (`--diarize` flag), you need a Hugging Face account and an access token (`hf_...`).
    *   Create an account at [huggingface.co](https://huggingface.co/).
    *   Generate an access token (read permissions are sufficient) in your account settings.
    *   **Crucially**, you must accept the user conditions for the Pyannote models *before* running the script with diarization for the first time:
        *   Go to: [hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) (Log in and accept terms)
        *   Go to: [hf.co/pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) (Log in and accept terms)
    *   You can provide the token via the `--hf_token YOUR_TOKEN` flag or by setting the `HF_TOKEN` environment variable.

6.  **Create the I/O folders**

    ```bash
    mkdir -p videos transcripts
    ```

## Usage

### Batch mode (recommended)

Simply drop one or more videos or audio files ( `.mp4 .mov .mkv .avi .mp3 .wav` ) into the `videos/` folder and run: