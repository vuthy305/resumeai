import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.services import get_manager, read_upload
from src.pipeline import PipelineError

router = APIRouter(tags=["extraction"])


@router.post("/extract")
async def extract(file: UploadFile = File(...)):
    data, original, ext = await read_upload(file)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{ext}"
        path.write_bytes(data)
        try:
            result = get_manager().pipeline.process_file(path)
        except PipelineError as e:
            raise HTTPException(422, str(e))
        except ValueError as e:  # corrupted PDF/DOCX/image
            raise HTTPException(422, str(e))
    result["filename"] = original
    return result
