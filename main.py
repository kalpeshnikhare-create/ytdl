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

# Used only by the /frames (testing) endpoint — fixed 6 positions as fractions of duration.
FRAME_POSITIONS = [0.05, 0.20, 0.35, 0.55, 0.75, 0.92]

# Maximum number of frames (seconds) sent to Claude in /analyze.
MAX_ANALYZE_FRAMES = 60

# Anthropic API key — set this as an environment variable in Render dashboard
# Render → Your Service → Environment → Add Environment Variable
# Key: ANTHROPIC_API_KEY   Value: sk-ant-...
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return JSONResponse(
            status_code=500,
            content={
                "error": "yt-dlp failed to download video",
                "stderr": result.stderr.decode(errors="replace")[-500:]
            }
        )

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
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status != 200:
                    raise ValueError(f"File server returned HTTP {response.status}")
                with open(video_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
        except urllib.error.HTTPError as e:
            raise ValueError(f"HTTPError fetching video file: {e.code} {e.reason} — file may have expired on server")
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


def _extract_frames(video_path: str, timestamps: list, frames_subdir: str) -> list:
    """
    Extract frames at the given list of timestamps (in seconds) and return list of dicts:
    { index, timestamp, data (base64), mediaType }
    """
    frames_out = []
    for i, timestamp in enumerate(timestamps):
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
            "data":      frame_b64,
            "mediaType": "image/jpeg"
        })
    return frames_out


@app.get("/frames")
def extract_frames_endpoint(url: str):
    """
    Download video and return base64 frames only (no Claude call).
    Uses fixed 6 positions for quick testing — kept unchanged.
    """
    video_path    = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4()}.mp4")
    frames_subdir = os.path.join(FRAMES_DIR, str(uuid.uuid4()))
    os.makedirs(frames_subdir, exist_ok=True)

    try:
        _download_video(url, video_path)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return JSONResponse(status_code=500,
                content={"error": "Failed to download video from the provided URL."})

        duration = _get_duration(video_path)

        # /frames still uses the original 6 fixed positions
        timestamps = [
            round(max(0.0, min(p * duration, duration - 0.1)), 1)
            for p in FRAME_POSITIONS
        ]
        frames_out = _extract_frames(video_path, timestamps, frames_subdir)

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
      2. Extract 1 frame per second (integer seconds: 0, 1, 2 … duration-1), capped at 60 frames
      3. Send frames as images to Claude
      4. Return structured analysis text

    `context` — optional account context string passed from GAS
                (URL-encoded, injected into the Claude prompt)

    Response:
    {
      "analysis":    "...",
      "frame_count": 55,
      "duration":    55,
      "warning":     null | "Video exceeds 60 seconds (Xs). Analysis is based on the first 60 seconds only."
    }
    """
    if not ANTHROPIC_API_KEY:
        return JSONResponse(status_code=500,
            content={"error": "ANTHROPIC_API_KEY environment variable is not set on the server."})

    video_path    = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4()}.mp4")
    frames_subdir = os.path.join(FRAMES_DIR,   str(uuid.uuid4()))
    os.makedirs(frames_subdir, exist_ok=True)

    try:
        # ── Step 1: Download ───────────────────────────────────────
        print(f"[ANALYZE] STEP 1 — downloading video from: {url[:100]}")
        _download_video(url, video_path)

        if not os.path.exists(video_path):
            print(f"[ANALYZE] STEP 1 FAIL — file does not exist after download: {video_path}")
            return JSONResponse(status_code=500,
                content={"error": "Failed to download video — file not created."})

        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        if file_size_mb == 0:
            print(f"[ANALYZE] STEP 1 FAIL — file exists but is 0 bytes")
            return JSONResponse(status_code=500,
                content={"error": "Downloaded video file is empty (0 bytes)."})

        print(f"[ANALYZE] STEP 1 OK — file size = {file_size_mb:.2f} MB")

        # ── Step 2: Get duration ───────────────────────────────────
        print(f"[ANALYZE] STEP 2 — probing video duration via ffprobe")
        duration = _get_duration(video_path)
        print(f"[ANALYZE] STEP 2 OK — duration = {duration:.1f}s")

        # ── Step 3: Compute per-second timestamps, cap at 60 ──────
        total_seconds = int(duration)  # e.g. 55.7s → 55
        capped        = total_seconds > MAX_ANALYZE_FRAMES
        num_frames    = min(total_seconds, MAX_ANALYZE_FRAMES)

        # Integer seconds: 0, 1, 2 … num_frames-1  (last = duration - 1 when uncapped)
        timestamps = list(range(0, num_frames))

        warning = None
        if capped:
            warning = (
                f"Video exceeds {MAX_ANALYZE_FRAMES} seconds "
                f"({total_seconds}s total). "
                f"Analysis is based on the first {MAX_ANALYZE_FRAMES} seconds only."
            )
            print(f"[ANALYZE] STEP 3 — {warning}")

        print(f"[ANALYZE] STEP 3 — extracting {num_frames} frames "
              f"(0s – {num_frames - 1}s, 1 frame/sec)")

        frames_out = _extract_frames(video_path, timestamps, frames_subdir)
        print(f"[ANALYZE] STEP 3 — extracted {len(frames_out)} frames "
              f"(expected {num_frames})")

        for f in frames_out:
            frame_kb = len(f['data']) * 3 / 4 / 1024
            print(f"[ANALYZE]   frame {f['index']} @ {f['timestamp']}s "
                  f"— base64 len={len(f['data'])} (~{frame_kb:.0f} KB decoded)")

        if not frames_out:
            print(f"[ANALYZE] STEP 3 FAIL — no frames extracted")
            return JSONResponse(status_code=500,
                content={"error": "ffmpeg could not extract any frames from this video. "
                                  "Check that ffmpeg is installed in the Docker image."})

        print(f"[ANALYZE] STEP 3 OK — {len(frames_out)} frames ready")

        # ── Step 4: Build Claude prompt ────────────────────────────
        print(f"[ANALYZE] STEP 4 — building Claude prompt")
        ctx_block = ""
        if context.strip():
            ctx_block = (
                "━━ ACCOUNT CONTEXT (factor this into your analysis) ━━\n"
                + context.strip() + "\n\n"
            )
            print(f"[ANALYZE] STEP 4 — context block length = {len(ctx_block)} chars")
        else:
            print(f"[ANALYZE] STEP 4 — no context provided")

        analyzed_duration = num_frames  # seconds covered (0 … num_frames-1)

        frame_labels = "\n".join(
            f"Frame {f['index'] + 1} → {f['timestamp']}s"
            for f in frames_out
        )

        analysis_prompt = (
            ctx_block +
            "You are a senior performance creative strategist for mCaffeine, a D2C skincare brand "
            "targeting primarily women aged 18–34 in India, running ads on Meta (Facebook & Instagram).\n\n"
            f"You have been given {len(frames_out)} frames extracted from a video ad at 1 frame per second "
            f"(Frame 1 = 0s, Frame {len(frames_out)} = {len(frames_out) - 1}s). "
            f"Total analyzed duration: {analyzed_duration} seconds.\n"
        )

        if warning:
            analysis_prompt += (
                f"NOTE: The original video is longer than {MAX_ANALYZE_FRAMES} seconds. "
                f"Your analysis covers only the first {MAX_ANALYZE_FRAMES} seconds.\n"
            )

        analysis_prompt += (
            f"\nFrame timestamps:\n{frame_labels}\n\n"
            "Each frame represents exactly 1 second of the video. Use the frame index and timestamp "
            "to reference specific moments precisely in your analysis.\n\n"
            "Perform a detailed analysis referencing what you actually see in specific frames — "
            "visuals, text overlays, talent, product shots, transitions, pacing, and messaging.\n\n"
            "Structure your response using EXACTLY these numbered headings:\n\n"
            "1. HOOK (0–3 SECONDS)\n"
            "Describe Frames 1–4 in detail. Is it thumb-stopping? Does it immediately create curiosity, "
            "emotion, or relevance? What technique is used (bold claim, question, visual contrast, etc.)?\n\n"
            "2. CREATIVE STRUCTURE\n"
            "Describe the overall narrative arc across all frames. Does it follow Problem → Solution, "
            "Before → After, Testimonial, Tutorial, or another format? "
            "How well does the structure serve the product?\n\n"
            "3. PACING & EDITING\n"
            "With 1 frame per second, comment on visual density, scene changes, text overlay frequency, "
            "and whether the pacing matches how the target audience consumes Reels/Feed content. "
            "Cite specific second-by-second transitions where relevant.\n\n"
            "4. MESSAGING & CTA\n"
            "What is the primary message? Is the value proposition clear and at which second does it appear? "
            "Identify the CTA frame — how is it framed and is it compelling?\n\n"
            "5. WHAT'S WORKING\n"
            "List exactly 3 specific creative elements that are effective, referencing the frame number "
            "and timestamp (in seconds).\n\n"
            "6. WHAT CAN BE IMPROVED\n"
            "List exactly 3 specific, actionable improvements with expected impact, referencing frames "
            "and timestamps (in seconds).\n\n"
            "7. OVERALL SCORE\n"
            "Rate this ad out of 10 for Meta performance potential. "
            "Give a single rationale sentence and one priority action for the creative team."
        )

        print(f"[ANALYZE] STEP 4 OK — prompt length = {len(analysis_prompt)} chars")

        # ── Step 5: Build content blocks ───────────────────────────
        print(f"[ANALYZE] STEP 5 — building {len(frames_out)} image content blocks")
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

        total_payload_kb = sum(len(f["data"]) for f in frames_out) / 1024
        print(f"[ANALYZE] STEP 5 OK — total image base64 payload = "
              f"{total_payload_kb:.0f} KB across {len(frames_out)} blocks")

        # ── Step 6: Call Claude ────────────────────────────────────
        print(f"[ANALYZE] STEP 6 — calling Claude API "
              f"(model=claude-sonnet-4-20250514, max_tokens=3500)")
        print(f"[ANALYZE] STEP 6 — ANTHROPIC_API_KEY set = "
              f"{'YES (len=' + str(len(ANTHROPIC_API_KEY)) + ')' if ANTHROPIC_API_KEY else 'NO — THIS WILL FAIL'}")

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        try:
            response = client.messages.create(
                model      = "claude-sonnet-4-20250514",
                max_tokens = 3500,
                messages   = [{"role": "user", "content": content_blocks}]
            )
            print(f"[ANALYZE] STEP 6 OK — response received")
            print(f"[ANALYZE] STEP 6 — stop_reason = {response.stop_reason}")
            print(f"[ANALYZE] STEP 6 — content blocks in response = {len(response.content)}")
        except anthropic.APIStatusError as api_err:
            print(f"[ANALYZE] STEP 6 FAIL — APIStatusError: "
                  f"status={api_err.status_code} body={str(api_err.body)[:400]}")
            return JSONResponse(status_code=500,
                content={"error": f"Claude API error {api_err.status_code}: {str(api_err.body)[:300]}"})
        except anthropic.APIConnectionError as conn_err:
            print(f"[ANALYZE] STEP 6 FAIL — connection error: {str(conn_err)}")
            return JSONResponse(status_code=500,
                content={"error": f"Claude connection error: {str(conn_err)}"})

        analysis_text = response.content[0].text if response.content else ""
        print(f"[ANALYZE] STEP 6 — analysis text length = {len(analysis_text)} chars")

        if not analysis_text:
            print(f"[ANALYZE] STEP 6 WARN — empty analysis text returned by Claude")

        print(f"[ANALYZE] COMPLETE — returning result to GAS")
        return JSONResponse({
            "analysis":    analysis_text,
            "frame_count": len(frames_out),
            "duration":    round(duration),
            "warning":     warning
        })

    except anthropic.APIError as e:
        print(f"[ANALYZE] EXCEPTION — anthropic.APIError: {str(e)}")
        return JSONResponse(status_code=500,
            content={"error": f"Claude API error: {str(e)}"})
    except subprocess.TimeoutExpired as te:
        print(f"[ANALYZE] EXCEPTION — TimeoutExpired: {str(te)}")
        return JSONResponse(status_code=504,
            content={"error": "Video processing timed out. The video may be too large."})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ANALYZE] EXCEPTION — {type(e).__name__}: {str(e)}")
        print(f"[ANALYZE] TRACEBACK:\n{tb}")
        return JSONResponse(status_code=500,
            content={"error": f"{type(e).__name__}: {str(e)}"})
    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"[ANALYZE] CLEANUP — removed video file")
        except Exception: pass
        try:
            shutil.rmtree(frames_subdir, ignore_errors=True)
            print(f"[ANALYZE] CLEANUP — removed frames directory")
        except Exception: pass
