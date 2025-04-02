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
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5178/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure upload directory
UPLOAD_DIR = Path("upload")
OUTPUT_DIR = Path("output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_unique_filename(original_name: str) -> str:
    """Generate a unique filename to prevent conflicts"""
    ext = Path(original_name).suffix
    return f"{uuid.uuid4().hex}{ext}"


@app.get("/")
def read_root():
    return {"message": "Video File Upload API"}


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Validate file extension
        allowed_extensions = {".mp4", ".mov", ".avi"}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Generate unique filename and save
        unique_filename = get_unique_filename(file.filename)
        file_path = UPLOAD_DIR / unique_filename
        output_path = OUTPUT_DIR / f"processed_{unique_filename}"

        logger.info(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the video
        logger.info(f"Starting video processing for: {file_path}")
        try:
            classifyShot(str(file_path), str(output_path))  # Modified to accept output path
        except Exception as processing_error:
            logger.error(f"Video processing failed: {str(processing_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Video processing error: {str(processing_error)}"
            )

        # Verify the output file exists
        if not output_path.exists():
            logger.error(f"Output file not found at: {output_path}")
            raise HTTPException(
                status_code=500,
                detail="Video processing failed - output file not created"
            )

        logger.info(f"Successfully processed video. Output at: {output_path}")

        # Return the processed video file
        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename=f"classified_{file.filename}"
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get-video/{filename}")
async def get_video(filename: str):
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if file_path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
        raise HTTPException(status_code=400, detail="Not a video file")

    return FileResponse(
        file_path,
        media_type="video/mp4" if file_path.suffix == ".mp4" else "video/quicktime",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=3000)