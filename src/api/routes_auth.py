from fastapi import APIRouter
from src.api.models import LoginRequest, LoginResponse
from src.auth.jwt import create_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

@router.post("/token", response_model=LoginResponse)
async def login(request: LoginRequest):
    token = create_token(username=request.username, groups=request.groups)
    return LoginResponse(access_token=token)
