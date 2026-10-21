import json
from pathlib import Path


def get_monitoring_status(system_name):
    # JSON 파일 읽기
    file_path = Path(__file__).with_name("monitoring.json")

    with file_path.open(encoding="utf-8") as f:
        monitoring = json.load(f)

    # 서비스 조회
    return monitoring.get(system_name)


# 사용자 입력
system_name = input("조회할 서비스 이름: ")

# 모니터링 조회
result = get_monitoring_status(system_name)

# 결과 출력
if result is None:
    print("등록되지 않은 서비스입니다.")
else:
    print(f"서비스: {system_name}")
    print(f"상태: {result['status']}")
    print(f"CPU: {result['cpu_base']}%")
    print(f"Memory: {result['memory_base']}%")
    print(f"알림: {result['alerts']}")