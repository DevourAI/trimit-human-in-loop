import os
import re
from pathlib import Path

from authlib.integrations.starlette_client import OAuthError
import authlib.jose.errors
from passlib.context import CryptContext
import aiofiles

from modal import asgi_app
from fastapi.responses import StreamingResponse
from fastapi import Depends, status, HTTPException, FastAPI, Request
from starlette.responses import RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from trimit.utils import conf
from trimit.app import app
from .image import image
from trimit.models import User, maybe_init_mongo
from trimit.utils.auth import (
    google_oauth,
    get_current_user,
    CREDENTIALS_EXCEPTION,
    create_token,
)
import secrets


ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # one week

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DynamicCORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        origin = request.headers.get("origin")
        # Regex to match allowed origins, e.g., any subdomain of trimit.vercel.app
        local_origins = ["http://127.0.0.1:3000", "http://localhost:3000"]
        allow_local = origin and origin in local_origins and os.environ["ENV"] == "dev"
        allow_remote = origin and re.match(r"https?://.*-trimit\.vercel\.app", origin)
        origin = origin or ""
        if allow_local or allow_remote:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range"
            )
            response.headers["Access-Control-Expose-Headers"] = (
                "Content-Length,Content-Range"
            )
        return response


TEMP_DIR = Path("/tmp/uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

web_app = FastAPI()
web_app.add_middleware(SessionMiddleware, secret_key=os.environ["AUTH_SECRET_KEY"])
FRONTEND_URL = os.environ["VERCEL_FRONTEND_URL"]
# TODO is this necessary?
web_app.add_middleware(DynamicCORSMiddleware)
#  allow_origins=[frontend_url],
#  allow_credentials=True,
#  allow_methods=["*"],  # Allow all methods
#  allow_headers=["*"],  # Allow all headers
#  )


app_kwargs = dict(
    _allow_background_volume_commits=True,
    timeout=80000,
    image=image,
    container_idle_timeout=1200,
    _experimental_boost=True,
    _experimental_scheduler=True,
)
oauth = google_oauth()


@app.function(**app_kwargs)
@asgi_app()
def frontend_server():
    return web_app


@web_app.get("/")
def public(request: Request):
    print("Received request")
    user = request.session.get("user")
    if user:
        name = user.get("name")
        email = user.get("email")
        return {"message": "logged in", "user_name": name, "user_email": email}
    return {"message": "logged out"}


@web_app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/")


@web_app.get("/login/google")
async def login_with_google(request: Request):
    redirect_uri = FRONTEND_URL
    # redirect_uri = request.url_for("authorize_with_google")

    state = secrets.token_urlsafe()
    request.session["oauth_state"] = state
    print("session in login", request.session)
    print("state in login", state)
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)


@web_app.get("/token")
async def authorize_with_google(request: Request):
    await maybe_init_mongo()
    expected_state = request.session.get("oauth_state")
    received_state = request.query_params.get("state")

    # Log both states
    print(f"Expected state from session: {expected_state}")
    print(f"Received state from Google: {received_state}")
    print("Cookie from client:", request.cookies.get("session"))

    print("session in authorize", request.session)
    try:
        access_token = await oauth.google.authorize_access_token(request)
    except OAuthError as e:
        print(f"Error authorizing access token: {e}")
        raise CREDENTIALS_EXCEPTION
    print("GOTG HERE")

    user_data = await oauth.google.parse_id_token(request, access_token)
    request.session["user"] = dict(user_data)

    email = user_data["email"]
    user = await User.find_one(User.email == email)
    if user is not None:
        user.authorized_with_google = True
    else:
        user = User(name=user_data["name"], email=email, authorized_with_google=True)
    await user.save()
    jwt = create_token(user.email)
    return JSONResponse(
        {
            "result": True,
            "access_token": jwt,
            "user_name": user_data["name"],
            "user_email": user_data["email"],
        }
    )


@web_app.get("/get_user_data")
async def get_user_data(current_user: User = Depends(get_current_user)):
    await maybe_init_mongo()
    return {"user_name": current_user.name, "user_email": current_user.email}


@web_app.post("/delete_account")
async def delete_account(current_user: User = Depends(get_current_user)):
    await maybe_init_mongo()
    await current_user.delete()
