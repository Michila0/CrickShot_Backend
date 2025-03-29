from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
from pathlib import Path

app = FastAPI()

# Configure upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist


@app.get("/")
def read_root():
    return {"message": "Video File Upload API"}


# Endpoint to upload video files
@app.post("/upload-video/")
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

        # Save file
        file_path = UPLOAD_DIR/file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()  # Read file content
            buffer.write(content)

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "filename": file.filename,
                "content_type": file.content_type
            }
        )

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

    uvicorn.run(app, host="0.0.0.0", port=3000)