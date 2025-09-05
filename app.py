from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from musicgen import MusicGen
import os
import uuid

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load MusicGen model (medium recommended for RunPod)
model = MusicGen.get_pretrained('facebook/musicgen-medium')

# Ensure folder exists
os.makedirs("generated", exist_ok=True)

class SongRequest(BaseModel):
    title: str
    lyrics: str = None

@app.post("/generate")
async def generate_song(request: SongRequest):
    prompt = request.lyrics if request.lyrics else request.title
    audio = model.generate(prompt=prompt, duration=180)  # 3 minutes
    file_id = str(uuid.uuid4())
    file_path = f"generated/{file_id}.wav"
    audio.save(file_path)
    return {"audio_url": f"/download/{file_id}.wav"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    return FileResponse(f"generated/{filename}")
