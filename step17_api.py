import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from langgraph.errors import GraphRecursionError

from step16_react import graph


# FastAPI 생성
app = FastAPI(title="AIOps Agent API")

logger = logging.getLogger(__name__)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"]
)


# 요청 데이터 검증
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("요청 내용이 비어 있습니다.")

        return value


# 응답 데이터 정의
class ChatResponse(BaseModel):
    intent: str
    result: str
    tasks: list[dict]


# Agent API
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    try:
        # LangGraph 실행
        state = await graph.ainvoke({
            "user_input": request.query
        })

        return ChatResponse(
            intent=state["intent"],
            result=state["result"],
            tasks=state.get("tasks", [])
        )

    # ReAct 반복 횟수 초과
    except GraphRecursionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Agent 실행 횟수 제한을 초과했습니다."
        ) from exc

    # 그 외 실행 오류
    except Exception as exc:
        logger.exception("Agent 실행 실패")

        raise HTTPException(
            status_code=500,
            detail="Agent 실행 중 오류가 발생했습니다."
        ) from exc