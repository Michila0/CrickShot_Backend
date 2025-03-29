# import os
# from fastapi import FastAPI, Request, Form, UploadFile, File
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse, StreamingResponse
# # from dotenv import load_dotenv
# # from supabase import create_client, Client
# import httpx
# from datetime import datetime, timezone
#
# # Load environment variables
# # load_dotenv()
#
# app = FastAPI()
#
# # Mount static files
# # app.mount("/static", StaticFiles(directory="static"), name="static")
#
# # Set up templates
# templates = Jinja2Templates(directory="templates")
#
# templates.env.globals.update(now=lambda: datetime.now(timezone.utc))
#
# # Initialize Supabase client
# # SUPABASE_URL = os.getenv('SUPABASE_URL')
# # SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
# # SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET')
#
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
#
# @app.get('/', response_class=HTMLResponse)
# async def home(request: Request):
#     videos = supabase.storage.from_(SUPABASE_BUCKET).list()
#     return templates.TemplateResponse('home.html', {'request': request, 'videos': videos})
#
# @app.get('/videos/{video_name}')
# async def get_video(video_name: str):
#     video_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(video_name)
#
#     if not video_url:
#         return {'error': 'video not found'}
#
#     async def video_stream():
#         async with httpx.AsyncClient() as client:
#             async with client.stream('GET', video_url, headers={'Range': 'bytes=0-'}, timeout=None) as response:
#                 async for chunck in response.aiter_bytes():
#                     yield chunck
#     return StreamingResponse(video_stream(), media_type='video/mp4')
#
#
# @app.get('/watch/{video_name}', response_class=HTMLResponse)
# async def watch_video(request: Request, video_name: str):
#     title = video_name.rsplit('.',1)[0].replace('_', ' ')
#     return templates.TemplateResponse('watch.html', {'request': request, 'video_name': video_name, 'title': title})
#
#
# @app.get('/upload', response_class=HTMLResponse)
# async def upload_form(request: Request):
#     return templates.TemplateResponse('upload.html', {'request': request})
#
# @app.post('/upload')
# async def upload_video(request: Request, title: str = File(...), video_file: UploadFile = File(...)):
#     contents = await video_file.read()
#
#     file_extension = video_file.filename.split('.')[-1]
#     file_name = f"{title.replace(' ', '_')}.{file_extension}"
#     res = supabase.storage.from_(SUPABASE_BUCKET).upload(file_name, contents)
#
#     if res.status_code >= 400:
#         message = 'Error uploading video.'
#     else:
#         message = 'Video uploaded successfully.'
#
#     return templates.TemplateResponse('upload.html', {'request': request, 'message': message})




# from fastapi import FastAPI
#
# app = FastAPI()
#
# @app.get("/")
# def read_root():
#     return {"message": "Hello, World!"}


from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

# Pydantic model for request body validation
class WordRequest(BaseModel):
    word: str

# File path (in the same directory)
FILE_PATH = os.path.join(os.path.dirname(__file__), "words.txt")


# Endpoint to handle POST request
@app.post("/add-word/")
async def add_word(word_request: WordRequest):
    word = word_request.word

    # Validate word is not empty
    if not word.strip():
        raise HTTPException(status_code=400, detail="Word cannot be empty")

    # Write word to file
    try:
        write_word_to_file(word)
        return JSONResponse(
            status_code=200,
            content={"message": f"Word '{word}' successfully written to file"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Function to write word to file
def write_word_to_file(word: str):
    file_path = os.path.join(os.path.dirname(__file__), "words.txt")

    # Open file in append mode (creates file if it doesn't exist)
    with open(file_path, "a") as file:
        file.write(word + "\n")


# GET endpoint: Returns the file
@app.get("/get-words/")
async def get_words():
    try:
        # Create file if it doesn't exist
        if not os.path.exists(FILE_PATH):
            open(FILE_PATH, "w").close()  # Create empty file

        return FileResponse(
            FILE_PATH,
            media_type="text/plain",  # Plain text file
            filename="words.txt"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For testing
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
