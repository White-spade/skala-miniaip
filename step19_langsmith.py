import asyncio
from dotenv import load_dotenv

# LangSmith 환경변수 로드
load_dotenv()

from langsmith import traceable
from step18_replan_validate import graph


# 사용자 요청 전체 추적
@traceable(name="aiops_chat_request")
async def run_agent(query: str):
    return await graph.ainvoke({"user_input": query})


async def main():
    query = input("요청: ")

    result = await run_agent(query)

    print("\n=== 최종 보고서 ===")
    print(result["result"])


if __name__ == "__main__":
    asyncio.run(main())