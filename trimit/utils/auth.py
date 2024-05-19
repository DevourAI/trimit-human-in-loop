from passlib.context import CryptContext
from authlib.integrations.starlette_client import OAuth
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
from datetime import datetime, timedelta, timezone
from trimit.models import User, maybe_init_mongo
import os
from functools import cache


async def authenticate_user(pwd_context: CryptContext, email: str, password: str):
    user = await User.find_one(User.email == email)
    if not user:
        return False
    if not pwd_context.verify(password, user.password):
        return False
    return user


def create_access_token(
    data: dict, expires_delta: timedelta | None = None, algorithm: str = "HS256"
):
    from jose import jwt

    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, os.environ["AUTH_SECRET_KEY"], algorithm=algorithm
    )
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme), algorithm: str = "HS256"
):
    from jose import JWTError, jwt
    from fastapi import HTTPException
    from fastapi import status

    await maybe_init_mongo()

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, os.environ["AUTH_SECRET_KEY"], algorithms=[algorithm]
        )
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await User.find_one(User.email == email)
    if user is None:
        raise credentials_exception
    return user


def google_oauth():
    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
        access_token_params=None,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        authorize_params=None,
        api_base_url="https://www.googleapis.com/oauth2/v1/",
        client_kwargs={"scope": "openid profile email"},
    )
    return oauth
