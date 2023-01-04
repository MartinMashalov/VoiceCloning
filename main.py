# main api endpoints
from fastapi import FastAPI, BackgroundTasks, Request, UploadFile, File
from schemas.response_schemas import MainVoiceClone
from typing import IO, Any, List
from backend import CustomMimic, GeneralMimic

# create app name
app = FastAPI()


@app.post('/predefined_voice_clone/', response_model=MainVoiceClone)
def predefined_voice_clone(specific_person: str, text_to_read: str) -> dict:
    """clone a predefined voice from a famous personality with the text intended to read"""
    pass

# create endpoint for main voice cloning functionality
@app.post('/custom_clone/', response_model=MainVoiceClone)
def custom_clone(text_to_read: str, files: List[UploadFile] = File(description="Multiple files as UploadFile")) -> dict:
    """voice cloning functionality: takes in file to clone voice from and text to synthesize"""

    # create class instance
    custom_voice_clone_backend: Any = CustomMimic(files)
    custom_voice_clone_backend.custom_voice_synthesize(text_to_read)
    return File()