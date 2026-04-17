from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import uuid
import os
import base64
import json
import shutil

app = FastAPI()

DOWNLOAD_DIR = "downloads"
FRAMES_DIR   = "frames"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR,   exist_ok=True)

BASE_URL = "https://instagram-downloader-qiht.onrender.com"

# Timestamps as fractions of total duration.
# Chosen to capture: hook, early body, mid, product zone, late body, CTA.
FRAME_POSITIONS = [0.05, 0.20, 0.35, 0.55, 0.75, 0.92]


@app.get("/")
def home():
    return {"status": "running"}


@app.get("/download")
def download(url: str):
    filename = f"{uuid.uuid4()}.mp4"
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    cmd = [
        "yt-dlp",
        "-f", "best",
        "-o", filepath,
        url
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return JSONResponse({
        "download_url": f"{BASE_URL}/file/{filename}"
    })


@app.get("/file/{filename}")
def get_file(filename: str):
    path = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/frames")
def extract_frames(url: str):
    """
    Download a video from `url`, extract 6 key frames using ffmpeg,
    and return them as base64-encoded JPEGs with timestamps.

    Response shape:
    {
      "frames": [
        {
          "index": 0,
          "timestamp": 0.5,
          "position": "5%",
          "data": "<base64 JPEG string>",
          "mediaType": "image/jpeg"
        },
        ...
      ],
      "duration": 15
    }
    """
    # ── Step 1: Download the video to a temp file ──────────────────
    video_filename = f"{uuid.uuid4()}.mp4"
    video_path     = os.path.join(DOWNLOAD_DIR, video_filename)
    frames_subdir  = os.path.join(FRAMES_DIR, str(uuid.uuid4()))
    os.makedirs(frames_subdir, exist_ok=True)

    try:
       # Use Python's urllib for direct MP4 links (no external tools needed).
        # Fall back to yt-dlp for Instagram/platform URLs.
        if url.endswith(".mp4") or "/file/" in url:
            import urllib.request
            urllib.request.urlretrieve(url, video_path)
        else:
            yt_cmd = ["yt-dlp", "-f", "best", "-o", video_path, url]
            subprocess.run(yt_cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=120)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to download video from the provided URL."}
            )

        # ── Step 2: Get video duration via ffprobe ─────────────────
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True,
                                      text=True, timeout=30)

        duration = 10.0   # fallback if probe fails
        try:
            probe_data = json.loads(probe_result.stdout)
            duration   = float(probe_data.get("format", {}).get("duration", 10.0))
        except Exception:
            pass  # use fallback duration

        # ── Step 3: Extract frames at each position ────────────────
        frames_out = []
        for i, position in enumerate(FRAME_POSITIONS):
            timestamp  = max(0.0, min(position * duration, duration - 0.1))
            timestamp  = round(timestamp, 1)
            frame_path = os.path.join(frames_subdir, f"frame_{i}.jpg")

            ffmpeg_cmd = [
                "ffmpeg",
                "-ss",       str(timestamp),   # seek BEFORE input for speed
                "-i",        video_path,
                "-frames:v", "1",              # extract exactly 1 frame
                "-vf",       "scale=720:-2",   # cap width at 720px, keep aspect ratio
                "-q:v",      "3",              # JPEG quality (1=best, 31=worst; 3 is good)
                "-y",                          # overwrite without prompting
                frame_path
            ]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)

            if not os.path.exists(frame_path):
                # Skip frames that failed to extract rather than aborting
                continue

            with open(frame_path, "rb") as f:
                frame_b64 = base64.b64encode(f.read()).decode("utf-8")

            frames_out.append({
                "index":     i,
                "timestamp": timestamp,
                "position":  f"{round(position * 100)}%",
                "data":      frame_b64,
                "mediaType": "image/jpeg"
            })

        if not frames_out:
            return JSONResponse(
                status_code=500,
                content={"error": "ffmpeg could not extract any frames from this video."}
            )

        return JSONResponse({
            "frames":   frames_out,
            "duration": round(duration)
        })

    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=504,
            content={"error": "Video processing timed out. The video may be too large."}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        # ── Cleanup temp files regardless of success or failure ─────
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass
        try:
            if os.path.exists(frames_subdir):
                shutil.rmtree(frames_subdir, ignore_errors=True)
        except Exception:
            pass
