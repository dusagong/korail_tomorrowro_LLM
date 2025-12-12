# DIGITS LLM Server

NVIDIA DIGITS (GB10)에서 실행되는 LLM 및 임베딩 서버입니다.

## 구성

| 서비스 | 모델 | 포트 |
|--------|------|------|
| LLM | EXAONE-3.5-32B-Instruct | 30000 |
| Embedding | BGE-M3 | 8080 |
| Vector DB | Qdrant | 6333 |

## 요구사항

- NVIDIA DIGITS (또는 119GB+ 통합 메모리 GPU)
- Docker + NVIDIA Container Toolkit
- HuggingFace Token

## 설치

```bash
# 1. 환경변수 설정
cp .env.example .env
# .env 파일에 HF_TOKEN 입력

# 2. Docker Compose 실행
docker compose up -d

# 3. 로그 확인
docker compose logs -f llm
```

## API 사용법

### Health Check
```bash
curl http://localhost:30000/health
```

### Chat Completion
```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "안녕하세요!"}],
    "max_tokens": 200
  }'
```

### Embedding
```bash
curl http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["안녕하세요", "Hello"]}'
```

## 라이선스

MIT