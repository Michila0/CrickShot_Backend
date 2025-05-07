from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from video_classifier import classifyShot
import os  # Keep this - it's actually used for path operations
import uuid
import logging
import mimetypes

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def get_unique_filename(original_name: str) -> str:
    """Generate a unique filename to prevent conflicts"""
    ext = Path(original_name).suffix
    return f"{uuid.uuid4().hex}{ext}"

@app.get("/")
def read_root():
    return {"message": "Video File Upload API"}

""""Stream Processing Pipline"""
async def stream_video(file_path: Path):
    """Generator function to stream video content"""
    with open(file_path, "rb") as video_file:
        while chunk := video_file.read(1024 * 1024):
            yield chunk

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

        # Generate unique filenames
        unique_filename = get_unique_filename(file.filename)
        file_path = UPLOAD_DIR / unique_filename
        output_path = OUTPUT_DIR / unique_filename

        # Save uploaded file
        logger.info(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)

        # Process the video
        logger.info(f"Starting video processing for: {file_path}")
        try:
            success = classifyShot(str(file_path), str(output_path))
            if not success:
                raise Exception("Video processing returned False")
        except Exception as processing_error:
            logger.error(f"Video processing failed: {str(processing_error)}")
            # Clean up files
            if file_path.exists():
                file_path.unlink()
            if output_path.exists():
                output_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"Video processing error: {str(processing_error)}"
            )

        # Verify output file
        if not output_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Video processing failed - output file not created"
            )

        logger.info(f"Successfully processed video. Output at: {output_path}")

        # Determine content type
        content_type, _ = mimetypes.guess_type(output_path)
        if content_type is None:
            content_type = "video/mp4"  # default to mp4

        # Return streaming response
        return StreamingResponse(
            stream_video(output_path),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=processed_{file.filename}",
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)