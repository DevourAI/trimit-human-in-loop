from passlib.context import CryptContext
from authlib.integrations.starlette_client import OAuth
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends
from fastapi import HTTPException
from fastapi import status
from jose import JWTError, jwt


from datetime import datetime, timedelta, timezone
from trimit.models import User, maybe_init_mongo
import os
from functools import cache


# Helper to read numbers using var envs
def cast_to_number(id):
    temp = os.environ.get(id)
    if temp is not None:
        try:
            return float(temp)
        except ValueError:
            return None
    return None


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
API_ALGORITHM = os.environ.get("API_ALGORITHM") or "HS256"
API_ACCESS_TOKEN_EXPIRE_MINUTES = (
    cast_to_number("API_ACCESS_TOKEN_EXPIRE_MINUTES") or 15
)
CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def create_token(email):
    access_token_expires = timedelta(minutes=API_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    return access_token


def create_access_token(
    data: dict, expires_delta: timedelta | None = None, algorithm: str = "HS256"
):
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
    await maybe_init_mongo()

    try:
        payload = jwt.decode(
            token, os.environ["AUTH_SECRET_KEY"], algorithms=[algorithm]
        )
        email: str = payload.get("sub")
        if email is None:
            raise CREDENTIALS_EXCEPTION
    except JWTError:
        raise CREDENTIALS_EXCEPTION
    user = await User.find_one(User.email == email)
    if user is None:
        raise CREDENTIALS_EXCEPTION
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
