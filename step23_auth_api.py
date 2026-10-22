"""STEP 23: FastAPI OAuth2 Password + JWT, with per-user SQLite sessions."""

import logging
import os
import re
import sqlite3
import sys
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta, timezone
from getpass import getpass

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator

load_dotenv()
from step21_sqlite_session import (  # noqa: E402
    SessionNotFoundError, connect, create_session, load_session, run_chat,
)


JWT_ALGORITHM = "HS256"
JWT_ISSUER = "ai-minip"
JWT_AUDIENCE = "ai-minip-api"
TOKEN_EXPIRE_MINUTES = 30
password_hasher = PasswordHasher()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def jwt_secret() -> str:
    secret = os.getenv("AIOPS_JWT_SECRET", "")
    if len(secret) < 32:
        raise RuntimeError(".env에 AIOPS_JWT_SECRET(32자 이상)을 설정하세요.")
    return secret


def init_auth_db():
    with closing(connect()) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS auth_users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_owners (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL
            )
        """)


def create_user(username: str, password: str):
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", username):
        raise ValueError("사용자명은 영문·숫자·_·.·-로 구성된 3~40자여야 합니다.")
    if len(password) < 8:
        raise ValueError("비밀번호는 최소 8자여야 합니다.")
    init_auth_db()
    with closing(connect()) as conn, conn:
        try:
            conn.execute(
                "INSERT INTO auth_users (username, password_hash) VALUES (?, ?)",
                (username, password_hasher.hash(password)),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("이미 존재하는 사용자입니다.") from exc


def lookup_user(username: str):
    with closing(connect()) as conn:
        return conn.execute(
            "SELECT username, password_hash FROM auth_users WHERE username = ?",
            (username,),
        ).fetchone()


def authenticate(username: str, password: str) -> bool:
    user = lookup_user(username)
    if user is None:
        return False
    try:
        return password_hasher.verify(user["password_hash"], password)
    except VerificationError:
        return False


def issue_token(username: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": username,
            "iat": now,
            "exp": now + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
            "iss": JWT_ISSUER,
            "aud": JWT_AUDIENCE,
        },
        jwt_secret(),
        algorithm=JWT_ALGORITHM,
    )


def current_user(token: str = Depends(oauth2_scheme)) -> str:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="유효한 인증 토큰이 필요합니다.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        claims = jwt.decode(
            token, jwt_secret(), algorithms=[JWT_ALGORITHM],
            issuer=JWT_ISSUER, audience=JWT_AUDIENCE,
            options={"require": ["sub", "iat", "exp", "iss", "aud"]},
        )
        username = claims["sub"]
        if not isinstance(username, str) or not lookup_user(username):
            raise unauthorized
    except jwt.InvalidTokenError as exc:
        raise unauthorized from exc
    return username


def check_owner(session_id: str, username: str):
    try:
        load_session(session_id)
    except (SessionNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="세션이 없습니다.") from exc
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT username FROM session_owners WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row is None or row["username"] != username:
        raise HTTPException(status_code=403, detail="이 세션에 접근할 권한이 없습니다.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    jwt_secret()
    init_auth_db()
    yield


app = FastAPI(title="STEP 23 AIOps OAuth2 + JWT", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    query: str = Field(min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("요청 내용이 비어 있습니다.")
        return value


@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if not authenticate(form.username, form.password):
        raise HTTPException(
            status_code=401, detail="아이디 또는 비밀번호가 잘못됐습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": issue_token(form.username), "token_type": "bearer"}


@app.get("/users/me")
def me(username: str = Depends(current_user)):
    return {"username": username}


@app.post("/session")
def new_session(username: str = Depends(current_user)):
    session_id = create_session()
    with closing(connect()) as conn, conn:
        conn.execute(
            "INSERT INTO session_owners (session_id, username) VALUES (?, ?)",
            (session_id, username),
        )
    return {"session_id": session_id}


@app.post("/chat")
async def chat(body: ChatRequest, username: str = Depends(current_user)):
    check_owner(body.session_id, username)
    try:
        return await run_chat(body.session_id, body.query)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="세션이 없습니다.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("AIOps 실행 오류")
        raise HTTPException(status_code=500, detail="Agent 실행 실패") from exc


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--create-user":
        print("사용법: python step23_auth_api.py --create-user 사용자명")
        sys.exit(1)
    try:
        password = getpass("새 비밀번호 (8자 이상): ")
        create_user(sys.argv[2], password)
        print(f"사용자 생성 완료: {sys.argv[2]}")
    except ValueError as exc:
        print(f"생성 실패: {exc}")
        sys.exit(1)
