from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import ffmpeg
import requests
import uuid
import shutil
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import os
from pathlib import Path
import video_classifier
from video_classifier import classifyShot

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5178/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Roboflow Configuration
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
ROBOFLOW_MODEL_ENDPOINT = f"https://detect.roboflow.com/crickshotsrilanka/2?api_key={ROBOFLOW_API_KEY}"

# Configure upload directory
UPLOAD_DIR = Path("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Create directory if it doesn't exist

#Extract Frames
def extract_frames(video_path: str, output_dir: str, frame_rate: int = 1) -> List[str]:
    """Extract frames from video using ffmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    try:
        (
            ffmpeg.input(video_path)
            .filter('fps', fps=frame_rate)
            .output(os.path.join(output_dir, 'frame_%04d.png'))
            .run(quiet=True)
        )
        frames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')])
    except ffmpeg.Error as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
    return frames

#Predict the Roboflow
def predict_with_roboflow(image_path: str) -> Dict:
    """Send image to Roboflow for prediction"""
    with open(image_path, "rb") as f:
        response = requests.post(
            ROBOFLOW_MODEL_ENDPOINT,
            files={"file": f},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    return response.json()


@app.get("/")
def read_root():
    return {"message": "Video File Upload API"}


# Endpoint to upload video files
@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Save uploaded video
        video_id = uuid.uuid4().hex
        video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract frames (1 frame per second)
        frames_dir = os.path.join(UPLOAD_DIR, video_id)
        frames = extract_frames(video_path, frames_dir, frame_rate=1)

        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted")

        # Process each frame
        predictions = []
        for frame_path in frames[:5]:  # Limit to 5 frames for demo
            prediction = predict_with_roboflow(frame_path)
            predictions.append({
                "frame": os.path.basename(frame_path),
                "prediction": prediction
            })

        classifyShot(video_path)
        # Cleanup
        shutil.rmtree(frames_dir)
        os.remove(video_path)

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to get video file
@app.get("/get-video/{filename}")
async def get_video(filename: str):
    file_path = UPLOAD_DIR / filename

    # Check if file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Check if file is video
    if file_path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
        raise HTTPException(status_code=400, detail="Not a video file")

    return FileResponse(
        file_path,
        media_type="video/mp4" if file_path.suffix == ".mp4" else "video/quicktime",
        filename=filename
    )


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=3000)