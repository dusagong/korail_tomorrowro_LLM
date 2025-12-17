# Tour MCP Server

한국관광공사 KorService2 API를 MCP 도구로 제공하는 서버

## 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 편집
vi .env
```

**.env 내용:**
```
KORSERVICE_URL=https://apis.data.go.kr/B551011/KorService2
TOUR_API_KEY=your_api_key_here
PORT=8000
```

## 실행 (DIGITS)

### 의존성 설치
```bash
cd tour-mcp-server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 포그라운드 실행
```bash
source venv/bin/activate
python server.py
```

### 백그라운드 실행
```bash
source venv/bin/activate
nohup python server.py > mcp.log 2>&1 &
echo $! > mcp.pid
```

## 종료 (DIGITS)

### PID 파일로 종료
```bash
kill $(cat mcp.pid)
rm mcp.pid
```

### 프로세스 찾아서 종료
```bash
# 프로세스 확인
ps aux | grep server.py

# 종료
kill <PID>

# 또는 포트로 찾아서 종료
lsof -ti:8000 | xargs kill -9
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| GET | `/api/tools` | 사용 가능한 도구 목록 |
| POST | `/api/call_tool` | 도구 호출 |

### 도구 호출 예시

```bash
# 키워드 검색
curl -X POST http://localhost:8000/api/call_tool \
  -H "Content-Type: application/json" \
  -d '{
    "name": "search_by_keyword",
    "arguments": {"keyword": "강릉", "content_type_id": "39"}
  }'

# 지역 기반 검색
curl -X POST http://localhost:8000/api/call_tool \
  -H "Content-Type: application/json" \
  -d '{
    "name": "search_by_area",
    "arguments": {"area_code": "32", "content_type_id": "39"}
  }'
```

## 사용 가능한 도구

| 도구명 | 설명 |
|--------|------|
| `get_area_codes` | 지역코드/시군구코드 조회 |
| `get_category_codes` | 서비스 분류코드 조회 |
| `search_by_area` | 지역기반 관광정보 검색 |
| `search_by_location` | GPS 위치기반 검색 |
| `search_by_keyword` | 키워드 검색 |
| `search_festivals` | 축제/행사 검색 |
| `search_stays` | 숙박 검색 |
| `get_detail_common` | 상세정보 (공통) |
| `get_detail_intro` | 상세정보 (소개) |
| `get_detail_info` | 상세정보 (반복) |
| `get_detail_images` | 이미지 목록 |
| `get_pet_tour_info` | 반려동물 여행정보 |

## 로그 확인

```bash
tail -f mcp.log
```
