from trimit.utils import conf
from trimit.app import app
from .image import image
from fastapi import FastAPI
from trimit.models import User, maybe_init_mongo
from fastapi import Request
from starlette.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuthError
from trimit.utils.auth import google_oauth, get_current_user
import aiofiles
from pathlib import Path
from fastapi.responses import StreamingResponse
import authlib.jose.errors
from passlib.context import CryptContext
from fastapi import Depends
from starlette.middleware.sessions import SessionMiddleware
from modal import asgi_app
import os


ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # one week

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


TEMP_DIR = Path("/tmp/uploads")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

web_app = FastAPI()
web_app.add_middleware(SessionMiddleware, secret_key=os.environ["AUTH_SECRET_KEY"])


app_kwargs = dict(
    _allow_background_volume_commits=True,
    timeout=80000,
    image=image,
    container_idle_timeout=1200,
    _experimental_boost=True,
    _experimental_scheduler=True,
)


@app.function(**app_kwargs)
@asgi_app()
def frontend_server():
    return web_app


@web_app.get("/api/")
def public(request: Request):
    user = request.session.get("user")
    if user:
        name = user.get("name")
        email = user.get("email")
        return {"user_name": name, "user_email": email}
    return {}


@web_app.route("/api/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/")


@web_app.get("/api/login/google")
async def login_with_google(request: Request):
    redirect_uri = request.url_for("authorize_with_google")
    return await google_oauth().google.authorize_redirect(request, redirect_uri)


@web_app.route("/api/auth/google")
async def authorize_with_google(request: Request):
    await maybe_init_mongo()
    try:
        access_token = await google_oauth().google.authorize_access_token(request)
    except OAuthError:
        print("Error in authorize with google", e)
        return RedirectResponse(url="/")
    user_data = await oauth.google.parse_id_token(request, access_token)
    request.session["user"] = dict(user_data)

    email = user_data["email"]
    user = await User.find_one(User.email == email)
    if user is not None:
        user.authorized_with_google = True
    else:
        user = User(name=user_data["name"], email=email, authorized_with_google=True)
    await user.save()
    return RedirectResponse(url="/")


@web_app.post("/api/delete_account")
async def delete_account(current_user: User = Depends(get_current_user)):
    await maybe_init_mongo()
    await current_user.delete()