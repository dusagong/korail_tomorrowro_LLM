# Korail TomorrowRo - 한국 여행 추천 시스템

AI 기반 한국 관광 여행 코스 추천 시스템입니다.
자연어로 여행 요청을 하면 최적의 여행 코스를 생성합니다.

## 프로젝트 구조

```
korail_tomorrowro/
├── llm_server_mcp.py      # 메인 서버 (LLM + MCP + 여행 추천)
├── llm_server.py          # 기본 LLM 서버
├── embedding_server.py    # 임베딩 서버
├── docker-compose.yml     # Docker 설정
├── .env.example           # 환경변수 예시
└── mcp_server/            # 관광 API MCP 서버 (독립 실행 가능)
    ├── server.py
    └── README.md
```

## 서비스 구성

| 서비스 | 설명 | 포트 |
|--------|------|------|
| LLM + MCP Server | EXAONE-3.5-32B + 여행 추천 | 30001 |
| MCP Server | 관광 API만 (경량) | 8000 |
| Qdrant | 벡터 DB (RAG용) | 6333 |

## 빠른 시작

### 1. 환경 설정

```bash
# 환경변수 파일 생성
cp .env.example .env

# .env 파일 편집
vi .env
```

**.env 내용:**
```
HF_TOKEN=your_huggingface_token
TOUR_API_KEY=your_tour_api_key
```

### 2. 실행

```bash
# 서비스 실행 (mcp-server + qdrant)
docker compose up -d

# LLM 서버도 함께 실행 (GPU 필요)
docker compose --profile mcp up -d

# 로그 확인
docker compose logs -f mcp-server
```

## API 사용법

### 여행 추천 요청 (메인 기능)

```bash
curl -X POST http://localhost:30001/v1/mcp/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "강릉에서 카페 갔다가 회 먹고 싶어"
  }'
```

**응답 예시:**
```json
{
  "spots": [...],
  "course": {
    "title": "강릉 카페 & 회 코스",
    "stops": [
      {"name": "테라로사 커피", "category": "카페", ...},
      {"name": "주문진 수산시장", "category": "횟집", ...}
    ],
    "total_distance_km": 15.2
  }
}
```

### 기타 API

```bash
# Health Check
curl http://localhost:30001/health

# Chat Completion (OpenAI 호환)
curl http://localhost:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "안녕하세요!"}],
    "max_tokens": 200
  }'

# MCP 도구 목록
curl http://localhost:30001/v1/mcp/tools
```

자세한 MCP Server 사용법: [mcp_server/README.md](mcp_server/README.md)

## 사용 가능한 도구

| 도구 | 설명 |
|------|------|
| `search_by_keyword` | 키워드로 관광지 검색 |
| `search_by_area` | 지역별 관광지 검색 |
| `search_by_location` | GPS 위치 기반 검색 |
| `search_festivals` | 축제/행사 검색 |
| `search_stays` | 숙박 검색 |
| `get_detail_common` | 상세정보 조회 |
| `get_detail_intro` | 소개정보 조회 |
| `get_detail_images` | 이미지 조회 |
| `get_area_codes` | 지역코드 조회 |
| `get_category_codes` | 카테고리 코드 조회 |
| `get_pet_tour_info` | 반려동물 여행정보 |

## 요구사항

- NVIDIA GPU (EXAONE-3.5-32B용, 119GB+ 메모리 권장)
- Docker + NVIDIA Container Toolkit
- HuggingFace Token (gated 모델 접근용)
- 한국관광공사 API Key

## 라이선스

MIT
