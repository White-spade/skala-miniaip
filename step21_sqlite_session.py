import asyncio
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from step20_clarification_hybrid import graph, mentioned_services, unknown_services


DB_PATH = Path(__file__).with_name("step21_sessions.db")
SESSION_FIELDS = (
    "pending_query", "pending_candidate", "clarification_attempts",
    "llm_clarification_used", "last_query", "last_targets",
)


class SessionNotFoundError(LookupError):
    pass


def empty_session() -> dict:
    return {
        "pending_query": "",
        "pending_candidate": "",
        "clarification_attempts": 0,
        "llm_clarification_used": False,
        "last_query": "",
        "last_targets": [],
    }


def connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with closing(connect()) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                pending_query TEXT NOT NULL DEFAULT '',
                pending_candidate TEXT NOT NULL DEFAULT '',
                clarification_attempts INTEGER NOT NULL DEFAULT 0,
                llm_clarification_used INTEGER NOT NULL DEFAULT 0,
                last_query TEXT NOT NULL DEFAULT '',
                last_targets TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def check_session_id(session_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise ValueError("유효하지 않은 session_id입니다.")


def load_session(session_id: str) -> dict:
    check_session_id(session_id)
    init_db()
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    if row is None:
        raise SessionNotFoundError(f"존재하지 않는 세션입니다: {session_id}")
    return {
        "pending_query": row["pending_query"],
        "pending_candidate": row["pending_candidate"],
        "clarification_attempts": row["clarification_attempts"],
        "llm_clarification_used": bool(row["llm_clarification_used"]),
        "last_query": row["last_query"],
        "last_targets": json.loads(row["last_targets"]),
    }


def save_session(session_id: str, session: dict):
    check_session_id(session_id)
    init_db()
    with closing(connect()) as conn, conn:
        conn.execute("""
            INSERT INTO sessions (
                session_id, pending_query, pending_candidate,
                clarification_attempts, llm_clarification_used,
                last_query, last_targets
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                pending_query = excluded.pending_query,
                pending_candidate = excluded.pending_candidate,
                clarification_attempts = excluded.clarification_attempts,
                llm_clarification_used = excluded.llm_clarification_used,
                last_query = excluded.last_query,
                last_targets = excluded.last_targets,
                updated_at = CURRENT_TIMESTAMP
        """, (
            session_id, session["pending_query"], session["pending_candidate"],
            session["clarification_attempts"], int(session["llm_clarification_used"]),
            session["last_query"], json.dumps(session["last_targets"], ensure_ascii=False),
        ))


def create_session() -> str:
    session_id = uuid4().hex
    save_session(session_id, empty_session())
    return session_id


def reset_session(session_id: str):
    load_session(session_id)
    save_session(session_id, empty_session())


def is_followup(query: str, session: dict) -> bool:
    if not session["last_targets"] or mentioned_services(query) or unknown_services(query):
        return False
    pattern = r"그\s*서비스|그럼|그러면|아까|방금|이어서|다시|추가로|해당|메모리|cpu|원인은|상태는"
    return re.search(pattern, query, flags=re.IGNORECASE) is not None


# 같은 세션의 동시 HTTP 요청이 상태를 덮어쓰지 않도록 직렬화
session_locks: dict[str, asyncio.Lock] = {}


async def run_chat(session_id: str, query: str) -> dict:
    check_session_id(session_id)
    query = query.strip()
    if not query or len(query) > 500:
        raise ValueError("요청은 1~500자여야 합니다.")

    lock = session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        if query == "/reset":
            reset_session(session_id)
            return {
                "session_id": session_id,
                "result": "세션을 초기화했습니다.",
                "awaiting_clarification": False,
                "clarification_attempts": 0,
                "last_targets": [],
            }

        session = load_session(session_id)
        followup = not session["pending_query"] and is_followup(query, session)
        effective_query = query
        if followup:
            effective_query += (
                f"\n이전 장애 요청: {session['last_query']}"
                f"\n조사 대상 서비스: {', '.join(session['last_targets'])}"
            )

        state = {key: session[key] for key in SESSION_FIELDS if key not in ("last_query", "last_targets")}
        result = await graph.ainvoke({**state, "user_input": effective_query})

        for key in ("pending_query", "pending_candidate", "clarification_attempts", "llm_clarification_used"):
            session[key] = result.get(key, session[key])

        if not session["pending_query"]:
            session.update({
                "pending_candidate": "",
                "clarification_attempts": 0,
                "llm_clarification_used": False,
            })
            if result.get("intent") == "diagnosis" and result.get("approved_targets") and "tasks" in result:
                session["last_targets"] = result["approved_targets"]
                if not followup:
                    session["last_query"] = result["user_input"]

        save_session(session_id, session)
        return {
            "session_id": session_id,
            "result": result.get("result", "응답이 없습니다."),
            "awaiting_clarification": bool(session["pending_query"]),
            "clarification_attempts": session["clarification_attempts"],
            "last_targets": session["last_targets"],
        }


async def main():
    init_db()
    supplied = input("기존 세션 ID (새 세션은 Enter): ").strip()
    if supplied:
        try:
            load_session(supplied)
        except (ValueError, SessionNotFoundError) as exc:
            print(f"세션 복원 실패: {exc}")
            return
        session_id = supplied
        print("기존 세션 복원 성공")
    else:
        session_id = create_session()
        print("새 세션 생성 완료")
    print(f"세션 ID: {session_id}")
    print("종료: /exit | 초기화: /reset | 상태 확인: /context")

    while True:
        query = input("\n요청: ").strip()
        if query == "/exit":
            break
        if query == "/context":
            session = load_session(session_id)
            print(json.dumps({"session_id": session_id, **session}, ensure_ascii=False, indent=2))
            continue
        if not query:
            continue
        response = await run_chat(session_id, query)
        print("\n" + response["result"])
        if response["awaiting_clarification"]:
            print(f"[서비스 확인 대기 / 오류 {response['clarification_attempts']}/3]")


if __name__ == "__main__":
    asyncio.run(main())
