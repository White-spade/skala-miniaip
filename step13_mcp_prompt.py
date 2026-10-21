import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    system_name = input("서비스 이름: ")
    symptom = input("장애 증상: ")

    server_path = Path(__file__).with_name("step13_mcp_server.py")

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)]
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 등록된 Prompt 목록 확인
            prompts = await session.list_prompts()
            print("사용 가능한 Prompt:", [p.name for p in prompts.prompts])

            # Prompt 요청
            response = await session.get_prompt(
                "incident_analysis_prompt",
                arguments={
                    "system_name": system_name,
                    "symptom": symptom
                }
            )

            # 반환된 메시지 출력
            for message in response.messages:
                if message.content.type == "text":
                    print("\n역할:", message.role)
                    print(message.content.text)


if __name__ == "__main__":
    asyncio.run(main())