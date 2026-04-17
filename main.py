from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
import subprocess
import uuid
import os

app = FastAPI()

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

BASE_URL = "https://instagram-downloader-qiht.onrender.com"

@app.get("/")
def home():
    return {"status": "running"}

from fastapi.responses import FileResponse

from fastapi.responses import FileResponse

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

    return FileResponse(
        filepath,
        media_type="video/mp4",
        filename="instagram_video.mp4"
    )
