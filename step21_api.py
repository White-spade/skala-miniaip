import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from step21_sqlite_session import SessionNotFoundError, create_session, run_chat


app = FastAPI(title="STEP 21 SQLite Session AIOps Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
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


@app.post("/session")
def new_session():
    return {"session_id": create_session()}


@app.post("/chat")
async def chat(body: ChatRequest):
    try:
        return await run_chat(body.session_id, body.query)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        logging.exception("Agent 실행 실패")
        raise HTTPException(status_code=500, detail="Agent 실행 중 오류가 발생했습니다.")
