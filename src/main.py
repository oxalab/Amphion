from fastapi import APIRouter, FastAPI

app = FastAPI()
router = APIRouter()

@router.get("/health")
async def health():
    return {"message": "The server is running healthy!"}

app.include_router(router)