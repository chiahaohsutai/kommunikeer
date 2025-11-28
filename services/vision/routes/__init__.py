from routes.analyze.router import router as analyze_router
from fastapi import APIRouter

router = APIRouter(prefix="/vision")
router.include_router(analyze_router)
