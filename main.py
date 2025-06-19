from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.params import Path as FastAPIPath  # Renamed to avoid conflict
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path  # This is for filesystem paths
import os
import uuid
import logging
import mimetypes
from video_classifier import classifyShot
from typing import List, Dict 
import re

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
    return f"{uuid.uuid4().hex}-{original_name}"

@app.get("/")
def read_root():
    return {"message": "Video File Upload API"}

def stream_video(file_path: Path):
    """Generator function to stream video content"""
    with open(file_path, "rb") as video_file:
        while chunk := video_file.read(1024 * 1024):
            yield chunk

@app.post("/upload-video/{uid}")
async def upload_video(
    file: UploadFile = File(...), 
    uid: str = FastAPIPath(...)  # Using the renamed FastAPI Path
):
    try:
        # Validate file extension
        allowed_extensions = {".mp4", ".mov", ".avi"}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )

        # Create UID-specific directories if they don't exist
        upload_dir = UPLOAD_DIR / uid
        output_dir = OUTPUT_DIR / uid
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filenames
        unique_filename = get_unique_filename(file.filename)
        file_path = upload_dir / unique_filename
        output_path = output_dir / unique_filename

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
    
@app.get("/videos/{uid}", response_model=List[Dict[str, str]])
def get_video_list(uid: str = FastAPIPath(...)):
    """
    Returns a list of videos for the given UID.
    Each video is represented as a JSON object with 'id' and 'name' fields.
    """
    try:
        # Get the output directory for this UID
        user_output_dir = OUTPUT_DIR / uid
        
        # Check if directory exists
        if not user_output_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No videos found for user {uid}"
            )
        
        video_files = []
        
        # Pattern to match the UUID prefix and capture both parts
        pattern = re.compile(r"^([a-f0-9]{32})-(.+\.mp4)$")
        
        # Scan through all files in the directory
        for file in user_output_dir.glob("*.mp4"):
            match = pattern.match(file.name)
            if match:
                video_id = match.group(1)
                video_name = match.group(2)
                video_files.append({
                    "id": video_id,
                    "name": video_name
                })
        
        if not video_files:
            raise HTTPException(
                status_code=404,
                detail=f"No valid video files found for user {uid}"
            )
            
        return video_files
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video list: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/videos/{uid}/{video_id}")
def get_video(
    uid: str = FastAPIPath(...),
    video_id: str = FastAPIPath(...)
):
    """
    Streams a specific video for the given UID and video ID.
    """
    try:
        # Get the output directory for this UID
        user_output_dir = OUTPUT_DIR / uid
        
        # Check if directory exists
        if not user_output_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"No videos found for user {uid}"
            )
        
        # Find the video file with the matching ID
        matching_files = list(user_output_dir.glob(f"{video_id}-*.mp4"))
        
        if not matching_files:
            raise HTTPException(
                status_code=404,
                detail=f"Video with ID {video_id} not found for user {uid}"
            )
        
        # There should only be one matching file
        video_path = matching_files[0]
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(video_path)
        if content_type is None:
            content_type = "video/mp4"  # default to mp4

        # Return streaming response
        return StreamingResponse(
            stream_video(video_path),
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename={video_path.name}",
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)