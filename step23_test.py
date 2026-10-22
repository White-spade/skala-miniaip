"""STEP 23 API 보안 테스트. Ollama/MCP 없이 실행합니다."""
import asyncio
import os
import sys
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 기존 Agent 대신 작은 모의 그래프 연결
stub = types.ModuleType("step20_clarification_hybrid")

class FakeGraph:
    async def ainvoke(self, state):
        return {
            **state,
            "intent": "chat",
            "result": "테스트 응답",
            "pending_query": "",
            "pending_candidate": "",
            "clarification_attempts": 0,
            "llm_clarification_used": False,
        }

stub.graph = FakeGraph()
stub.mentioned_services = lambda query: []
stub.unknown_services = lambda query: []
sys.modules["step20_clarification_hybrid"] = stub

import step21_sqlite_session as session  # noqa: E402
import step23_auth_api as auth  # noqa: E402


def check(label, cond):
    assert cond, label
    print("PASS", label)


def main():
    with tempfile.TemporaryDirectory() as temp:
        session.DB_PATH = Path(temp) / "test_sessions.db"
        os.environ["AIOPS_JWT_SECRET"] = "step23_testing_only_super_secret_0123456789"
        session.init_db()
        auth.create_user("alice", "testpassword123")
        auth.create_user("bob", "anotherpassword123")
        with TestClient(auth.app) as client:
            check("미인증 /session 차단", client.post("/session").status_code == 401)
            check("틀린 비밀번호 차단", client.post("/token", data={"username": "alice", "password": "wrong"}).status_code == 401)
            good = client.post("/token", data={"username": "alice", "password": "testpassword123"})
            check("정상 로그인", good.status_code == 200)
            headers = {"Authorization": "Bearer " + good.json()["access_token"]}
            check("JWT 사용자 확인", client.get("/users/me", headers=headers).json() == {"username": "alice"})
            check("변조 토큰 차단", client.get("/users/me", headers={"Authorization": "Bearer " + good.json()["access_token"][:-2] + "xx"}).status_code == 401)

            expired = jwt.encode({
                "sub":"alice", "iss":auth.JWT_ISSUER, "aud":auth.JWT_AUDIENCE,
                "iat":datetime.now(timezone.utc) - timedelta(hours=1),
                "exp":datetime.now(timezone.utc) - timedelta(minutes=1)
            }, auth.jwt_secret(), algorithm="HS256")
            check("만료된 JWT 차단", client.get("/users/me", headers={"Authorization":"Bearer "+expired}).status_code == 401)

            r = client.post("/session", headers=headers)
            check("내 세션 생성", r.status_code == 200)
            sid = r.json()["session_id"]
            data = {"session_id":sid,"query":"/reset"}
            check("소유자 /chat 허용", client.post("/chat", json=data, headers=headers).status_code == 200)

            bob = client.post("/token", data={"username":"bob", "password":"anotherpassword123"})
            bob_headers = {"Authorization":"Bearer " + bob.json()["access_token"]}
            check("다른 사용자 세션 차단", client.post("/chat", json=data, headers=bob_headers).status_code == 403)
            legacy_id = session.create_session()
            check("미귀속 기존 세션 차단", client.post("/chat", json={"session_id":legacy_id,"query":"hi"}, headers=headers).status_code == 403)
            check("없는 세션 404", client.post("/chat", json={"session_id":"0"*32,"query":"hi"}, headers=headers).status_code == 404)
            check("정상 Agent 요청", client.post("/chat", json={"session_id":sid,"query":"hello"}, headers=headers).json()["result"] == "테스트 응답")
            check("공백 입력 422", client.post("/chat", json={"session_id":sid,"query":"   "}, headers=headers).status_code == 422)

    print("STEP23 핵심 보안 테스트 통과")


if __name__ == "__main__":
    main()
