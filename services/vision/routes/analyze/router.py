from fastapi import APIRouter, UploadFile, HTTPException, Depends
from mimetypes import guess_type
from typing import Annotated
from models.ocr import TextExtractor


_accepted = frozenset(("image/jpeg", "image/png", "image/jpg"))
router = APIRouter(prefix="/analyze")


def _validate_file(file: UploadFile):
    """Validate the uploaded file's content type and extension."""

    ctype = file.content_type
    mime = ctype if ctype else guess_type(str(file.filename))[0]

    if mime not in _accepted:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    return file


@router.post("/")
def analyze(file: Annotated[UploadFile, Depends(_validate_file)]):
    """Analyze the uploaded file and return its extension."""

    result = TextExtractor()(file.file.read())
    return result
