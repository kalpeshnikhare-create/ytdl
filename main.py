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
from transcriber import transcribe_with_timestamps

app = FastAPI()

COOKIES_PATH = "/tmp/instagram_cookies.txt"

# Auto-update yt-dlp on startup
try:
    update_result = subprocess.run(
        ["yt-dlp", "-U"],
        capture_output=True, text=True, timeout=60
    )
    print(f"[STARTUP] yt-dlp update: {update_result.stdout.strip() or update_result.stderr.strip()}")
except Exception as e:
    print(f"[STARTUP] yt-dlp update failed: {e}")

# Write Instagram cookies from environment variable to disk
_cookies_env = os.environ.get("INSTAGRAM_COOKIES", "")
if _cookies_env:
    with open(COOKIES_PATH, "w") as _f:
        _f.write(_cookies_env)
    print(f"[STARTUP] Instagram cookies written to {COOKIES_PATH}")
else:
    print(f"[STARTUP] WARNING — INSTAGRAM_COOKIES env var not set. Downloads may fail.")

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
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")

if not OPENAI_API_KEY:
    print("[STARTUP] WARNING — OPENAI_API_KEY not set. Audio transcription will be skipped.")
    
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
        "--cookies", COOKIES_PATH,
        "--extractor-args", "instagram:app=android",
        "--no-check-certificates",
        "--user-agent", "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36",
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
        yt_cmd = [
            "yt-dlp",
            "-f", "best",
            "--cookies", COOKIES_PATH,
            "--extractor-args", "instagram:app=android",
            "--no-check-certificates",
            "--user-agent", "Mozilla/5.0 (Linux; Android 11; SM-G991B) AppleWebKit/537.36",
            "-o", video_path,
            url
        ]
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


def _extract_audio(video_path: str, audio_path: str) -> bool:
    """
    Extract audio from video file using ffmpeg as 16kHz mono WAV.
    WAV/PCM needs no external codec, works on all ffmpeg builds, and is Whisper's native rate.
    Returns True if audio file was created and is non-empty, False otherwise.
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                   # strip video
        "-acodec", "pcm_s16le",  # PCM — no external codec needed
        "-ar", "16000",          # 16kHz — Whisper's native sample rate
        "-ac", "1",              # mono
        "-y",                    # overwrite
        audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        print(f"[AUDIO] ffmpeg extraction failed.\n"
              f"[AUDIO] Command: {' '.join(cmd)}\n"
              f"[AUDIO] stderr: {result.stderr.decode(errors='replace')[-500:]}")
        return False
    size_kb = os.path.getsize(audio_path) / 1024
    print(f"[AUDIO] Extracted audio — {size_kb:.1f} KB at 16kHz mono WAV")
    return True


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

        # ── Step 3b: Extract audio + transcribe ───────────────────
        transcript_data   = None
        transcript_block  = ""
        audio_path        = os.path.splitext(video_path)[0] + ".wav"

        print(f"[ANALYZE] STEP 3b — extracting audio for transcription")
        try:
            audio_ok = _extract_audio(video_path, audio_path)
            if audio_ok:
                print(f"[ANALYZE] STEP 3b — audio extracted, calling Whisper API")
                transcript_data = transcribe_with_timestamps(audio_path)
                print(f"[ANALYZE] STEP 3b OK — transcript length = "
                      f"{len(transcript_data.get('full_text', ''))} chars, "
                      f"{len(transcript_data.get('second_by_second', []))} seconds mapped")

                # Format transcript for Claude prompt
                sbs = transcript_data.get("second_by_second", [])
                if sbs:
                    sbs_lines = "\n".join(
                        f"  {entry['second']}s: {entry['text']}"
                        for entry in sbs
                    )
                    transcript_block = (
                        "━━ AUDIO TRANSCRIPT (second-by-second) ━━\n"
                        f"Full text: {transcript_data.get('full_text', '').strip()}\n\n"
                        f"Timestamped breakdown:\n{sbs_lines}\n\n"
                    )
                else:
                    transcript_block = (
                        "━━ AUDIO TRANSCRIPT ━━\n"
                        f"{transcript_data.get('full_text', '').strip()}\n\n"
                    )
            else:
                print(f"[ANALYZE] STEP 3b WARN — audio extraction failed, proceeding without transcript")
                transcript_block = "━━ AUDIO TRANSCRIPT ━━\n[Audio could not be extracted from this video]\n\n"
        except Exception as te:
            print(f"[ANALYZE] STEP 3b WARN — transcription failed: {te}. Proceeding without transcript.")
            transcript_block = f"━━ AUDIO TRANSCRIPT ━━\n[Transcription failed: {str(te)}]\n\n"
        finally:
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass

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
            transcript_block +
            "You are a senior performance creative strategist for mCaffeine, a D2C skincare brand "
            "targeting primarily women aged 18–44 in India, running ads on Meta (Facebook & Instagram).\n\n"
            f"You have been given {len(frames_out)} frames extracted from a video ad at 1 frame per second "
            f"(Frame 1 = 0s, Frame {len(frames_out)} = {len(frames_out) - 1}s). "
            f"Total analyzed duration: {analyzed_duration} seconds.\n"
            "You also have a second-by-second audio transcript of the same video (provided above). "
            "Cross-reference what is SAID (transcript) with what is SHOWN (frames) at each moment — "
            "alignment or misalignment between audio and visuals is a key creative signal.\n"
        )

        if warning:
            analysis_prompt += (
                f"NOTE: The original video is longer than {MAX_ANALYZE_FRAMES} seconds. "
                f"Your analysis covers only the first {MAX_ANALYZE_FRAMES} seconds.\n"
            )

        analysis_prompt += (
            f"\nFrame timestamps:\n{frame_labels}\n\n"
            "Each frame represents exactly 1 second of the video. "
            "Where the transcript provides spoken words at that second, quote them directly in your analysis. "
            "Cross-reference WHAT IS SAID vs WHAT IS SHOWN at every key moment — "
            "alignment or contradiction between audio and visuals is a critical performance signal.\n\n"
            "Analyse this ad as a performance creative strategist whose job is to predict and improve "
            "Meta (Facebook + Instagram) ad performance. Go beyond describing what you see — "
            "interpret WHY each creative choice does or does not work for a paid media context.\n\n"
            "Structure your response using EXACTLY these numbered headings:\n\n"

            "1. HOOK STRENGTH (0–3s)\n"
            "Analyse Frames 1–4 alongside the transcript for those seconds.\n"
            "— Pattern interrupt: Does Frame 1 stop a thumb mid-scroll? What visual device is used "
            "(movement, face, bold text, unexpected image, colour contrast)?\n"
            "— Audio hook: What are the first spoken words or sounds? Do they create a loop, question, "
            "or bold claim that demands attention?\n"
            "— Audio-visual sync: Is the spoken hook reinforced or contradicted by what is shown?\n"
            "— Hook archetype: Label it — Bold Claim / Pain Point / Curiosity Gap / Social Proof Open / "
            "Contrast / Story Open / Direct Address / Shock Visual.\n"
            "— Predicted 3-second hold rate: Strong / Average / Weak — with one-line justification.\n\n"

            "2. RETENTION & PACING\n"
            "Map the likely audience drop-off curve against the frame sequence.\n"
            "— Identify the single highest drop-off risk second (where attention is most likely lost) "
            "and explain what visual or audio element causes it.\n"
            "— Identify the strongest re-engagement moment (a scene change, reveal, or spoken line "
            "that pulls back a distracted viewer) and cite the exact second.\n"
            "— Scene density: How many distinct visual scenes appear across all frames? "
            "Is the cut rate too slow (boring), too fast (disorienting), or well-matched to Reels consumption?\n"
            "— Text overlay pacing: Are on-screen captions or text cards timed so a viewer can read "
            "them before the scene cuts? Cite any seconds where text and cut timing clash.\n\n"

            "3. NARRATIVE & CREATIVE STRUCTURE\n"
            "— Label the creative format: UGC Testimonial / Founder Story / Problem→Solution / "
            "Before→After / Tutorial / Social Proof Stack / Product Demo / Talking Head / "
            "Cinematic Brand / Trend-Jacked / Other.\n"
            "— Map the narrative arc second-by-second at a high level: "
            "what does the viewer understand at 0s, at the midpoint, and at the final second?\n"
            "— Identify where the emotional peak sits (the moment of maximum curiosity, desire, or "
            "relief) — cite the second and what creates it visually and aurally.\n"
            "— Does the spoken script and the visual story tell the SAME story, or do they operate "
            "independently? Independent audio and video is a common conversion killer — flag it if present.\n\n"

            "4. VALUE PROPOSITION & MESSAGING CLARITY\n"
            "— What is the single core claim this ad makes? State it in one sentence.\n"
            "— At which exact second does the viewer first understand what the product is and why they need it?\n"
            "— Is the claim specific and verifiable (e.g. '2% salicylic acid', 'dermatologist tested') "
            "or vague (e.g. 'makes skin glow')? Quote the exact spoken or on-screen words.\n"
            "— Ingredient / benefit callout: Is the hero ingredient or product benefit shown on-screen, "
            "spoken, or both? At which second?\n"
            "— Is there a clear reason-to-believe (RTB) — before/after, expert endorsement, ingredient "
            "proof, user count, certifications? Cite the frame.\n"
            "— Fear of Missing Out or urgency signal: Is there any scarcity, limited-time, or "
            "social-proof element that compresses the decision to buy? If absent, flag it.\n\n"

            "5. TALENT & AUTHENTICITY SIGNALS\n"
            "— Is there an on-screen person (founder, influencer, customer, actor)? "
            "If yes: does their presentation feel native-UGC or produced-ad? "
            "Native UGC consistently outperforms polished production for Meta DTC.\n"
            "— Eye contact & direct address: Does the talent look directly into the camera "
            "at the hook? Direct-address hooks have higher thumb-stop rates — note if this is used.\n"
            "— Skin & product demonstration: Is the product shown being used on actual skin? "
            "At which second? Demonstration frames are among the highest-converting visual elements "
            "for skincare — flag if this is missing or occurs too late.\n"
            "— Trust cues: Are certifications, dermatologist mentions, or 'real customer' signals "
            "visible on screen or spoken? At which second do they appear?\n\n"

            "6. AUDIO STRATEGY\n"
            "— Background music: Does it match the emotional tone of the visuals? "
            "Is it trending audio (which boosts organic reach on Reels) or generic?\n"
            "— Voiceover vs. on-camera speech: Which is used? "
            "Voiceover allows faster information delivery; on-camera speech builds trust — "
            "is the right choice made for this ad's goal?\n"
            "— Sound-off watchability: Can a viewer understand the full message with sound off "
            "(via text overlays and visual storytelling alone)? "
            "~85% of Meta feed ads are watched without sound — flag any seconds where the message "
            "is ONLY conveyed through audio with no visual reinforcement.\n"
            "— Spoken CTA: Is the CTA spoken aloud? At which second? "
            "Spoken + visual CTA together outperform either alone.\n\n"

            "7. CALL TO ACTION (CTA)\n"
            "— At which exact second does the CTA appear (visual, spoken, or both)?\n"
            "— CTA type: Direct purchase / Link in bio / Shop now / Learn more / DM us / Other.\n"
            "— Friction assessment: Does the CTA require the viewer to do something hard "
            "(find a link, remember a code) or is it frictionless (tap Shop Now)?\n"
            "— Is there an incentive attached to the CTA (discount, free shipping, bundle)? "
            "If absent, flag it as a missed conversion opportunity.\n"
            "— Is the CTA visually prominent — large text, contrasting colour, long enough on screen "
            "to read and act? Or does it flash past in under 1 second?\n\n"

            "8. PLATFORM & FORMAT FIT\n"
            "— Aspect ratio & framing: Is the video shot for vertical (9:16) Reels/Stories, "
            "square (1:1) Feed, or horizontal (16:9 repurposed)? "
            "Repurposed horizontal video consistently underperforms native vertical.\n"
            "— Safe zone compliance: Is any key text or product shot cut off by Instagram UI elements "
            "(bottom bar, profile name, caption)? Cite the second if so.\n"
            "— Native content feel: Does this look like something a friend would post, "
            "or does it immediately read as an ad? Ads that feel native have lower CPMs and higher CTR.\n"
            "— Hook-to-brand reveal timing: How many seconds before the brand name or product appears? "
            "Revealing brand too early triggers skip; too late loses attribution — flag the timing.\n\n"

            "9. WHAT'S WORKING (exactly 3 points)\n"
            "For each point: name the element → cite the exact frame and second → "
            "explain the performance mechanism (why it works for Meta DTC, not just why it looks good).\n\n"

            "10. WHAT TO FIX (exactly 3 points)\n"
            "For each point: name the problem → cite the exact frame and second → "
            "give a specific, implementable fix → state the expected performance impact "
            "(e.g. 'should improve 3-second hold rate', 'likely to increase CTR', "
            "'reduces cognitive load at decision point').\n\n"

            "11. CREATIVE SCORECARD\n"
            "Score each dimension out of 10. Be ruthlessly honest — a 7 should feel earned.\n"
            "  Hook Strength:         /10\n"
            "  Retention & Pacing:    /10\n"
            "  Messaging Clarity:     /10\n"
            "  Product Demonstration: /10\n"
            "  Audio-Visual Sync:     /10\n"
            "  CTA Effectiveness:     /10\n"
            "  Platform Fit:          /10\n"
            "  OVERALL SCORE:         /10\n\n"
            "Overall verdict: one sentence on Meta performance potential.\n"
            "Single highest-priority action for the creative team to implement before the next run."
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

@app.get("/test-audio")
def test_audio(url: str):
    """
    Diagnostic endpoint — tests only audio extraction + transcription.
    Skips frames and Claude entirely. Use to verify Whisper pipeline in isolation.
    """
    video_path = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4()}.mp4")
    audio_path = os.path.splitext(video_path)[0] + ".wav"

    results = {
        "step1_download":    None,
        "step2_audio":       None,
        "step3_transcript":  None,
        "error":             None
    }

    try:
        # Step 1: Download
        print(f"[TEST-AUDIO] Downloading: {url[:100]}")
        _download_video(url, video_path)

        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            results["step1_download"] = "FAIL — video not downloaded"
            return JSONResponse(results)

        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        results["step1_download"] = f"OK — {size_mb:.2f} MB"
        print(f"[TEST-AUDIO] Download OK — {size_mb:.2f} MB")

        # Step 2: Extract audio
        audio_ok = _extract_audio(video_path, audio_path)

        if not audio_ok:
            results["step2_audio"] = "FAIL — ffmpeg could not extract audio"
            return JSONResponse(results)

        size_kb = os.path.getsize(audio_path) / 1024
        results["step2_audio"] = f"OK — {size_kb:.1f} KB WAV at 16kHz mono"
        print(f"[TEST-AUDIO] Audio OK — {size_kb:.1f} KB")

        # Step 3: Transcribe
        transcript = transcribe_with_timestamps(audio_path)

        results["step3_transcript"] = {
            "status":           "OK",
            "full_text":        transcript.get("full_text", ""),
            "segment_count":    len(transcript.get("segments", [])),
            "seconds_mapped":   len(transcript.get("second_by_second", [])),
            "second_by_second": transcript.get("second_by_second", [])
        }
        print(f"[TEST-AUDIO] Transcript OK — {len(transcript.get('full_text', ''))} chars")

    except Exception as e:
        import traceback
        results["error"] = f"{type(e).__name__}: {str(e)}"
        print(f"[TEST-AUDIO] Exception: {traceback.format_exc()}")

    finally:
        for path in [video_path, audio_path]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    return JSONResponse(results)
