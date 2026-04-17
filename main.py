from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import uuid
import os
import base64
import json
import shutil
import urllib.request
import anthropic

app = FastAPI()

DOWNLOAD_DIR = "downloads"
FRAMES_DIR   = "frames"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR,   exist_ok=True)

BASE_URL = "https://instagram-downloader-qiht.onrender.com"

# Timestamps as fractions of total duration.
# Chosen to capture: hook, early body, mid, product zone, late body, CTA.
FRAME_POSITIONS = [0.05, 0.20, 0.35, 0.55, 0.75, 0.92]

# Anthropic API key — set this as an environment variable in Render dashboard
# Render → Your Service → Environment → Add Environment Variable
# Key: ANTHROPIC_API_KEY   Value: sk-ant-...
ANTHROPIC_API_KEY = os.environ.get("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", "")


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


def _download_video(url: str, video_path: str):
    """Download a video to video_path. Uses urllib for direct MP4s, yt-dlp for platform URLs."""
    if url.endswith(".mp4") or "/file/" in url:
        urllib.request.urlretrieve(url, video_path)
    else:
        yt_cmd = ["yt-dlp", "-f", "best", "-o", video_path, url]
        subprocess.run(yt_cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=120)


def _get_duration(video_path: str) -> float:
    """Return video duration in seconds via ffprobe."""
    probe_cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
    try:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 10.0))
    except Exception:
        return 10.0


def _extract_frames(video_path: str, duration: float, frames_subdir: str) -> list:
    """
    Extract frames at FRAME_POSITIONS and return list of dicts:
    { index, timestamp, position, data (base64), mediaType }
    """
    frames_out = []
    for i, position in enumerate(FRAME_POSITIONS):
        timestamp  = round(max(0.0, min(position * duration, duration - 0.1)), 1)
        frame_path = os.path.join(frames_subdir, f"frame_{i}.jpg")

        ffmpeg_cmd = [
            "ffmpeg",
            "-ss",       str(timestamp),
            "-i",        video_path,
            "-frames:v", "1",
            "-vf",       "scale=720:-2",
            "-q:v",      "3",
            "-y",
            frame_path
        ]
        subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=30)

        if not os.path.exists(frame_path):
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
    return frames_out


@app.get("/frames")
def extract_frames_endpoint(url: str):
    """
    Download video and return base64 frames only (no Claude call).
    Kept for testing purposes.
    """
    video_path    = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4()}.mp4")
    frames_subdir = os.path.join(FRAMES_DIR, str(uuid.uuid4()))
    os.makedirs(frames_subdir, exist_ok=True)

    try:
        _download_video(url, video_path)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return JSONResponse(status_code=500,
                content={"error": "Failed to download video from the provided URL."})

        duration   = _get_duration(video_path)
        frames_out = _extract_frames(video_path, duration, frames_subdir)

        if not frames_out:
            return JSONResponse(status_code=500,
                content={"error": "ffmpeg could not extract any frames."})

        return JSONResponse({"frames": frames_out, "duration": round(duration)})

    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504,
            content={"error": "Video processing timed out."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            if os.path.exists(video_path):    os.remove(video_path)
        except Exception: pass
        try:
            shutil.rmtree(frames_subdir, ignore_errors=True)
        except Exception: pass


@app.get("/analyze")
def analyze_video(url: str, context: str = ""):
    """
    Full pipeline in one endpoint:
      1. Download video from `url`
      2. Extract 6 frames via ffmpeg
      3. Send frames as images to Claude
      4. Return structured analysis text

    `context` — optional account context string passed from GAS
                (URL-encoded, injected into the Claude prompt)

    Response: { "analysis": "...", "frame_count": 6, "duration": 15 }
    """
    if not ANTHROPIC_API_KEY:
        return JSONResponse(status_code=500,
            content={"error": "ANTHROPIC_API_KEY environment variable is not set on the server."})

    video_path    = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4()}.mp4")
    frames_subdir = os.path.join(FRAMES_DIR,   str(uuid.uuid4()))
    os.makedirs(frames_subdir, exist_ok=True)

    try:
        # ── Step 1: Download ───────────────────────────────────────
        _download_video(url, video_path)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return JSONResponse(status_code=500,
                content={"error": "Failed to download video from the provided URL."})

        # ── Step 2: Extract frames ─────────────────────────────────
        duration   = _get_duration(video_path)
        frames_out = _extract_frames(video_path, duration, frames_subdir)

        if not frames_out:
            return JSONResponse(status_code=500,
                content={"error": "ffmpeg could not extract any frames from this video."})

        # ── Step 3: Build Claude prompt ────────────────────────────
        ctx_block = ""
        if context.strip():
            ctx_block = (
                "━━ ACCOUNT CONTEXT (factor this into your analysis) ━━\n"
                + context.strip() + "\n\n"
            )

        frame_labels = "\n".join(
            f"Frame {f['index'] + 1} → {f['timestamp']}s ({f['position']} through video)"
            for f in frames_out
        )

        analysis_prompt = (
            ctx_block +
            "You are a senior performance creative strategist for mCaffeine, a D2C skincare brand "
            "targeting primarily women aged 18–34 in India, running ads on Meta (Facebook & Instagram).\n\n"
            f"You have been given {len(frames_out)} key frames extracted from a video ad "
            f"(total duration: {round(duration)} seconds). "
            f"Frame timestamps:\n{frame_labels}\n\n"
            "Perform a detailed frame-by-frame analysis. Reference what you actually see in specific "
            "frames — visuals, text overlays, talent, product shots, transitions, pacing, and messaging.\n\n"
            "Structure your response using EXACTLY these numbered headings:\n\n"
            "1. HOOK (0–3 SECONDS)\n"
            "Describe Frame 1 in detail. Is it thumb-stopping? Does it immediately create curiosity, "
            "emotion, or relevance? What technique is used (bold claim, question, visual contrast, etc.)?\n\n"
            "2. CREATIVE STRUCTURE\n"
            "Describe the overall narrative arc across all frames. Does it follow Problem → Solution, "
            "Before → After, Testimonial, Tutorial, or another format? "
            "How well does the structure serve the product?\n\n"
            "3. PACING & EDITING\n"
            "Based on what you see across frames, comment on visual density, text overlay frequency, "
            "and whether the pacing matches how the target audience consumes Reels/Feed content.\n\n"
            "4. MESSAGING & CTA\n"
            "What is the primary message? Is the value proposition clear and when does it appear? "
            "Identify the CTA frame — how is it framed and is it compelling?\n\n"
            "5. WHAT'S WORKING\n"
            "List exactly 3 specific creative elements that are effective, referencing the frame number "
            "and timestamp.\n\n"
            "6. WHAT CAN BE IMPROVED\n"
            "List exactly 3 specific, actionable improvements with expected impact, referencing frames.\n\n"
            "7. OVERALL SCORE\n"
            "Rate this ad out of 10 for Meta performance potential. "
            "Give a single rationale sentence and one priority action for the creative team."
        )

        # ── Step 4: Build image content blocks for Claude ──────────
        content_blocks = [
            {
                "type":   "image",
                "source": {
                    "type":       "base64",
                    "media_type": f["mediaType"],
                    "data":       f["data"]
                }
            }
            for f in frames_out
        ]
        content_blocks.append({"type": "text", "text": analysis_prompt})

        # ── Step 5: Call Claude ────────────────────────────────────
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 1800,
            messages   = [{"role": "user", "content": content_blocks}]
        )

        analysis_text = response.content[0].text if response.content else ""

        return JSONResponse({
            "analysis":    analysis_text,
            "frame_count": len(frames_out),
            "duration":    round(duration)
        })

    except anthropic.APIError as e:
        return JSONResponse(status_code=500,
            content={"error": f"Claude API error: {str(e)}"})
    except subprocess.TimeoutExpired:
        return JSONResponse(status_code=504,
            content={"error": "Video processing timed out. The video may be too large."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            if os.path.exists(video_path):    os.remove(video_path)
        except Exception: pass
        try:
            shutil.rmtree(frames_subdir, ignore_errors=True)
        except Exception: pass
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
