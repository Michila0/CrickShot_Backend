from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.params import Path as FastAPIPath
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import uuid
import logging
import mimetypes
from typing import List, Dict, Optional
import re

import mysql.connector
from mysql.connector import Error

# Assume video_classifier module exists and has classifyShot function
try:
    from video_classifier import classifyShot
except ImportError:
    # Fallback for local development if video_classifier isn't present
    logging.warning("video_classifier.py not found. Using a dummy classifyShot function.")
    def classifyShot(input_path: str, output_path: str) -> bool:
        """Dummy function for video processing."""
        logging.info(f"Dummy processing: Copying {input_path} to {output_path}")
        try:
            # Simulate processing by copying the file
            with open(input_path, "rb") as infile, open(output_path, "wb") as outfile:
                outfile.write(infile.read())
            return True
        except Exception as e:
            logging.error(f"Dummy processing failed: {e}")
            return False

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development, restrict in production
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

# Global MySQL Database Configuration
# IMPORTANT: Replace with your actual MySQL credentials
DB_CONFIG = {
    'host': 'localhost',
    'database': 'CricketVideo', # Ensure this database exists and has the 'videos' table
    'user': 'root',
    'password': '' # Add your MySQL password here if you have one
}

# --- MySQL Database Manager Class ---
class MySQLVideoManager:
    def __init__(self, host, database, user, password):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None

    def connect(self) -> bool:
        """Establishes a connection to the MySQL database."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                logger.info(f"Successfully connected to MySQL database: {self.database}")
                return True
            else:
                logger.error("Failed to establish a connection to MySQL.")
                return False
        except Error as e:
            logger.error(f"Error while connecting to MySQL: {e}")
            self.connection = None # Ensure connection is None on failure
            return False

    def disconnect(self):
        """Closes the database connection if it's open."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("MySQL connection closed.")
        else:
            logger.debug("No active MySQL connection to close.")

    def add_video(self, uid: str, original_filename: str, unique_filename: str, uploaded_filepath: Path, processed_filepath: Path) -> Optional[int]:
        """
        Adds a new video record to the 'videos' table with both original and unique names,
        and paths for uploaded and processed files.

        Args:
            uid (str): The user ID associated with the video.
            original_filename (str): The original name of the file provided by the user.
            unique_filename (str): The internally generated unique filename (UUID-original_name.ext).
            uploaded_filepath (Path): Absolute path to the original uploaded file.
            processed_filepath (Path): Absolute path to the processed output file.

        Returns:
            int or None: The ID of the newly inserted record if successful, otherwise None.
        """
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected to the database. Cannot add video metadata.")
            return None

        # Store the unique_filename in video_name and processed_filepath in video_url
        sql = "INSERT INTO videos (uid, video_name, video_url) VALUES (%s, %s, %s)"
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, (uid, unique_filename, str(processed_filepath)))
            self.connection.commit()
            logger.info(f"Video metadata for '{original_filename}' (Unique: {unique_filename}) added to DB with ID: {cursor.lastrowid}")
            return cursor.lastrowid
        except Error as e:
            logger.error(f"Error adding video metadata for '{original_filename}': {e}")
            self.connection.rollback()
            return None
        finally:
            if 'cursor' in locals() and cursor is not None:
                cursor.close()

    def get_videos_by_uid(self, uid: str) -> List[Dict]:
        """
        Retrieves all video records associated with a specific user ID.

        Args:
            uid (str): The user ID.

        Returns:
            List[Dict]: A list of dictionaries, each representing a video record.
        """
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected to the database. Cannot retrieve videos.")
            return []

        sql = "SELECT id, video_name, video_url, created_at FROM videos WHERE uid = %s"
        videos = []
        try:
            cursor = self.connection.cursor(dictionary=True) # Return rows as dictionaries
            cursor.execute(sql, (uid,))
            videos = cursor.fetchall()
            return videos
        except Error as e:
            logger.error(f"Error retrieving videos for UID '{uid}': {e}")
            return []
        finally:
            if 'cursor' in locals() and cursor is not None:
                cursor.close()

    def get_video_by_db_id(self, db_video_id: int, uid: str) -> Optional[Dict]:
        """
        Retrieves a single video record by its database ID and UID.

        Args:
            db_video_id (int): The primary key ID of the video in the database.
            uid (str): The user ID to ensure ownership.

        Returns:
            Optional[Dict]: A dictionary representing the video record, or None if not found.
        """
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected to the database. Cannot retrieve video by ID.")
            return None

        sql = "SELECT id, video_name, video_url, uid FROM videos WHERE id = %s AND uid = %s"
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(sql, (db_video_id, uid))
            video = cursor.fetchone()
            return video
        except Error as e:
            logger.error(f"Error retrieving video by DB ID {db_video_id} for UID {uid}: {e}")
            return None
        finally:
            if 'cursor' in locals() and cursor is not None:
                cursor.close()

    def delete_video_by_db_id(self, db_video_id: int, uid: str) -> bool:
        """
        Deletes a video record from the database by its primary key ID and UID.

        Args:
            db_video_id (int): The primary key ID of the video in the database.
            uid (str): The user ID to ensure the video belongs to this user.

        Returns:
            bool: True if the record was deleted, False otherwise.
        """
        if not self.connection or not self.connection.is_connected():
            logger.error("Not connected to the database. Cannot delete video metadata.")
            return False

        sql = "DELETE FROM videos WHERE id = %s AND uid = %s"
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, (db_video_id, uid))
            self.connection.commit()
            if cursor.rowcount > 0:
                logger.info(f"Video metadata with DB ID {db_video_id} for UID {uid} deleted successfully.")
                return True
            else:
                logger.warning(f"No video metadata found with DB ID {db_video_id} for UID {uid} or it doesn't belong to this user.")
                return False
        except Error as e:
            logger.error(f"Error deleting video metadata with DB ID {db_video_id}: {e}")
            self.connection.rollback()
            return False
        finally:
            if 'cursor' in locals() and cursor is not None:
                cursor.close()

# --- Helper Functions for FastAPI ---
def get_unique_filename(original_name: str) -> str:
    """Generate a unique filename (UUID-original_name) to prevent conflicts"""
    return f"{uuid.uuid4().hex}-{original_name}"

def stream_video(file_path: Path):
    """Generator function to stream video content"""
    # Ensure the file exists and is readable before attempting to open
    if not file_path.is_file() or not os.access(file_path, os.R_OK):
        logger.error(f"File not found or not readable: {file_path}")
        raise FileNotFoundError(f"File not found or not accessible: {file_path}")

    try:
        with open(file_path, "rb") as video_file:
            while chunk := video_file.read(1024 * 1024): # 1MB chunks
                yield chunk
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading video file: {e}")


# --- FastAPI Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Video File Upload and Management API"}

@app.post("/upload-video/{uid}")
async def upload_video(
    file: UploadFile = File(...),
    uid: str = FastAPIPath(..., description="Unique user identifier")
):
    """
    Uploads a video file, saves it, processes it, stores metadata in MySQL,
    and returns the processed video as a stream.
    """
    if not uid:
        raise HTTPException(status_code=400, detail="UID cannot be empty.")

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
    original_filename = file.filename
    unique_filename = get_unique_filename(original_filename)
    uploaded_filepath = upload_dir / unique_filename
    processed_filepath = output_dir / unique_filename # Processed file will have the same unique name

    # Initialize DB Manager and connect
    manager = MySQLVideoManager(**DB_CONFIG)
    if not manager.connect():
        raise HTTPException(
            status_code=500,
            detail="Failed to connect to the database. Please check server status and credentials."
        )

    try:
        # Save uploaded file
        logger.info(f"Saving uploaded file to: {uploaded_filepath}")
        with open(uploaded_filepath, "wb") as buffer:
            # Read and write in chunks to handle large files
            while chunk := await file.read(1024 * 1024): # 1MB chunks
                buffer.write(chunk)
        logger.info(f"File '{original_filename}' saved to {uploaded_filepath}")

        # Process the video
        logger.info(f"Starting video processing for: {uploaded_filepath}")
        try:
            # classifyShot should handle creating the output_path file
            success = classifyShot(str(uploaded_filepath), str(processed_filepath))
            if not success:
                raise Exception("Video processing returned False or encountered an error.")
        except Exception as processing_error:
            logger.error(f"Video processing failed: {str(processing_error)}")
            # Clean up uploaded file if processing fails
            if uploaded_filepath.exists():
                uploaded_filepath.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"Video processing error: {str(processing_error)}"
            )

        # Verify output file exists after processing
        if not processed_filepath.exists():
            # This should ideally be caught by classifyShot's error handling
            logger.error(f"Processed output file not found at: {processed_filepath}")
            raise HTTPException(
                status_code=500,
                detail="Video processing failed - output file was not created or is missing."
            )

        logger.info(f"Successfully processed video. Output at: {processed_filepath}")

        # Store video metadata in MySQL after successful save and processing
        db_record_id = manager.add_video(
            uid=uid,
            original_filename=original_filename,
            unique_filename=unique_filename,
            uploaded_filepath=uploaded_filepath,
            processed_filepath=processed_filepath
        )

        if db_record_id is None:
            # If DB storage fails, log it and possibly clean up files
            logger.error(f"Failed to save video metadata to database for {original_filename}. Cleaning up files.")
            if uploaded_filepath.exists():
                uploaded_filepath.unlink()
            if processed_filepath.exists():
                processed_filepath.unlink()
            raise HTTPException(
                status_code=500,
                detail="Failed to store video metadata in the database."
            )

        # Determine content type for streaming
        content_type, _ = mimetypes.guess_type(processed_filepath)
        if content_type is None:
            content_type = "video/mp4" # Default to mp4 if type cannot be guessed

        # Return streaming response of the processed video
        return StreamingResponse(
            stream_video(processed_filepath),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=processed_{original_filename}",
                "Accept-Ranges": "bytes", # Essential for video streaming/seeking
                "Cache-Control": "no-cache"
            }
        )

    except HTTPException:
        # Re-raise HTTPExceptions as they are intended responses
        raise
    except FileNotFoundError as e:
        logger.error(f"File system error: {e}")
        raise HTTPException(status_code=500, detail="Server file system error.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        # Ensure database connection is always closed
        manager.disconnect()

@app.get("/videos/{uid}", response_model=List[Dict[str, str]])
async def get_video_list(
    uid: str = FastAPIPath(..., description="Unique user identifier")
):
    """
    Retrieves a list of video metadata (ID and original filename) for the given UID from the database.
    The 'id' in the response refers to the UUID part of the filename, used for streaming.
    """
    if not uid:
        raise HTTPException(status_code=400, detail="UID cannot be empty.")

    manager = MySQLVideoManager(**DB_CONFIG)
    if not manager.connect():
        raise HTTPException(
            status_code=500,
            detail="Failed to connect to the database. Please check server status and credentials."
        )

    try:
        db_videos = manager.get_videos_by_uid(uid)
        if not db_videos:
            raise HTTPException(
                status_code=404,
                detail=f"No videos found for user '{uid}' in the database."
            )

        video_list_response = []
        # Pattern to match the UUID prefix and capture the original filename
        pattern = re.compile(r"^([a-f0-9]{32})-(.+)$", re.IGNORECASE)

        for video_record in db_videos:
            # video_name in DB stores 'UUID-original_filename.ext'
            match = pattern.match(video_record['video_name'])
            if match:
                # The 'id' in the response will be the UUID part for consistency with file-based lookup
                video_uuid = match.group(1)
                # The 'name' in the response will be the part after UUID (original_filename.ext)
                original_display_name = match.group(2)
                video_list_response.append({
                    "id": video_uuid,
                    "name": original_display_name
                    # Note: We are not returning the full processed_filepath here for security/API simplicity
                })
            else:
                logger.warning(f"Could not parse unique filename: {video_record['video_name']}. Skipping.")

        if not video_list_response:
             raise HTTPException(
                status_code=404,
                detail=f"No parsable video files found for user '{uid}' in the database."
            )

        return video_list_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video list for UID {uid}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        manager.disconnect()


@app.get("/videos/{uid}/{video_id}")
async def get_video_stream(
    uid: str = FastAPIPath(..., description="Unique user identifier"),
    video_id: str = FastAPIPath(..., description="UUID part of the video filename")
):
    """
    Streams a specific processed video file from disk, identified by UID and its UUID filename.
    This endpoint directly serves the processed video file from the OUTPUT_DIR.
    """
    if not uid or not video_id:
        raise HTTPException(status_code=400, detail="UID and Video ID cannot be empty.")

    # Construct the potential file path based on common pattern
    # We are using glob to find the exact filename as original_name part might vary
    user_output_dir = OUTPUT_DIR / uid
    matching_files = list(user_output_dir.glob(f"{video_id}-*"))

    if not matching_files:
        logger.warning(f"Video file not found for UID '{uid}' with UUID '{video_id}' in directory {user_output_dir}.")
        raise HTTPException(
            status_code=404,
            detail=f"Video with ID '{video_id}' not found for user '{uid}'."
        )

    # Assuming only one file matches the UUID prefix
    video_path = matching_files[0]

    if not video_path.is_file():
        logger.error(f"Identified path is not a file: {video_path}")
        raise HTTPException(status_code=500, detail="Invalid video file path on server.")

    # Determine content type
    content_type, _ = mimetypes.guess_type(video_path)
    if content_type is None:
        content_type = "video/mp4" # Default if mimetype cannot be guessed

    try:
        return StreamingResponse(
            stream_video(video_path),
            media_type=content_type,
            headers={
                "Content-Disposition": f"inline; filename={video_path.name}", # 'inline' to play in browser
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache"
            }
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Video file not found or inaccessible.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error streaming video {video_id} for UID {uid}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred during streaming: {str(e)}")


@app.delete("/videos/db/{uid}/{db_video_id}")
async def delete_video_record(
    uid: str = FastAPIPath(..., description="Unique user identifier"),
    db_video_id: int = FastAPIPath(..., description="Database primary key ID of the video record")
):
    """
    Deletes a video record from the database by its primary key ID and also deletes
    the associated original uploaded file and processed output file from the filesystem.
    """
    if not uid or not db_video_id:
        raise HTTPException(status_code=400, detail="UID and DB Video ID cannot be empty.")

    manager = MySQLVideoManager(**DB_CONFIG)
    if not manager.connect():
        raise HTTPException(
            status_code=500,
            detail="Failed to connect to the database. Please check server status and credentials."
        )

    try:
        # First, retrieve the record to get file paths for deletion
        video_record = manager.get_video_by_db_id(db_video_id, uid)
        if not video_record:
            raise HTTPException(
                status_code=404,
                detail=f"Video record with DB ID {db_video_id} not found for user '{uid}'."
            )

        unique_filename = video_record['video_name'] # This is 'UUID-original_name.ext'
        processed_filepath_from_db = Path(video_record['video_url']) # This should be absolute path

        # Construct uploaded file path - assuming same unique_filename in uploads
        uploaded_filepath = UPLOAD_DIR / uid / unique_filename

        # Delete the record from the database
        if not manager.delete_video_by_db_id(db_video_id, uid):
            raise HTTPException(status_code=500, detail="Failed to delete video metadata from database.")

        # Delete associated files from the filesystem
        deleted_files_count = 0
        if processed_filepath_from_db.exists():
            try:
                processed_filepath_from_db.unlink()
                deleted_files_count += 1
                logger.info(f"Deleted processed file: {processed_filepath_from_db}")
            except OSError as e:
                logger.error(f"Error deleting processed file {processed_filepath_from_db}: {e}")

        if uploaded_filepath.exists():
            try:
                uploaded_filepath.unlink()
                deleted_files_count += 1
                logger.info(f"Deleted uploaded file: {uploaded_filepath}")
            except OSError as e:
                logger.error(f"Error deleting uploaded file {uploaded_filepath}: {e}")

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Video record (DB ID: {db_video_id}) and {deleted_files_count} associated files deleted successfully."
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video with DB ID {db_video_id} for UID {uid}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")
    finally:
        manager.disconnect()

# --- Uvicorn Run Configuration ---
if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn your_script_name:app --reload --port 5000
    uvicorn.run(app, host="0.0.0.0", port=5000)
