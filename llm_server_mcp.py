"""
EXAONE-3.5-32B Server with MCP Host functionality
LLM이 MCP 도구를 선택하고 호출하는 기능 포함
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import httpx
import json
import asyncio
import os
from typing import Optional

app = FastAPI(title="EXAONE-3.5-32B Server + MCP Host")

# ========== MCP Server 설정 ==========
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")

# ========== 모델 로딩 ==========
print("Loading model...")
model_id = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_memory={0: "100GiB", "cpu": "50GiB"},
    offload_buffers=True,
)

device = next(model.parameters()).device
print(f"Model loaded! Device: {device}")

# ========== MCP 도구 정의 (tour-mcp-server 기준) ==========
MCP_TOOLS = [
    {
        "name": "get_area_codes",
        "description": "지역코드/시군구코드 목록 조회",
        "parameters": {"area_code": "선택 - 지역코드"}
    },
    {
        "name": "search_by_area",
        "description": "지역기반 관광정보 검색. 특정 지역의 관광지, 음식점, 숙박 등 조회",
        "parameters": {
            "area_code": "필수 - 지역코드",
            "content_type_id": "선택 - 콘텐츠타입",
            "sigungu_code": "선택 - 시군구코드",
            "num_of_rows": "선택 - 결과개수"
        }
    },
    {
        "name": "search_by_keyword",
        "description": "키워드로 관광정보 검색. 가장 유연한 검색 방법",
        "parameters": {
            "keyword": "필수 - 검색키워드",
            "area_code": "선택 - 지역코드",
            "content_type_id": "선택 - 콘텐츠타입",
            "num_of_rows": "선택 - 결과개수"
        }
    },
    {
        "name": "search_by_location",
        "description": "GPS 위치 기반 주변 관광정보 검색",
        "parameters": {
            "map_x": "필수 - 경도",
            "map_y": "필수 - 위도",
            "radius": "선택 - 반경(미터)",
            "content_type_id": "선택 - 콘텐츠타입"
        }
    },
    {
        "name": "search_festivals",
        "description": "축제/행사 정보 검색",
        "parameters": {
            "event_start_date": "필수 - 시작일(YYYYMMDD)",
            "event_end_date": "선택 - 종료일(YYYYMMDD)",
            "area_code": "선택 - 지역코드"
        }
    },
    {
        "name": "search_stays",
        "description": "숙박 정보 검색",
        "parameters": {
            "area_code": "선택 - 지역코드",
            "sigungu_code": "선택 - 시군구코드"
        }
    },
    {
        "name": "get_detail_common",
        "description": "관광지 상세정보 조회 (주소, 이미지, 개요 등)",
        "parameters": {
            "content_id": "필수 - 콘텐츠ID",
            "content_type_id": "필수 - 콘텐츠타입"
        }
    },
    {
        "name": "get_detail_intro",
        "description": "관광지 소개정보 조회 (운영시간, 입장료 등)",
        "parameters": {
            "content_id": "필수 - 콘텐츠ID",
            "content_type_id": "필수 - 콘텐츠타입"
        }
    },
    {
        "name": "get_detail_images",
        "description": "관광지 이미지 목록 조회",
        "parameters": {"content_id": "필수 - 콘텐츠ID"}
    },
    {
        "name": "get_category_codes",
        "description": "서비스 분류코드 조회",
        "parameters": {
            "content_type_id": "선택 - 콘텐츠타입",
            "cat1": "선택 - 대분류",
            "cat2": "선택 - 중분류"
        }
    },
    {
        "name": "get_detail_info",
        "description": "소개정보 조회 (반복정보)",
        "parameters": {
            "content_id": "필수 - 콘텐츠ID",
            "content_type_id": "필수 - 콘텐츠타입"
        }
    },
    {
        "name": "get_pet_tour_info",
        "description": "반려동물 여행정보 조회",
        "parameters": {
            "area_code": "선택 - 지역코드",
            "sigungu_code": "선택 - 시군구코드"
        }
    }
]

# 지역코드 매핑
AREA_CODES = {
    "서울": "1", "인천": "2", "대전": "3", "대구": "4", "광주": "5",
    "부산": "6", "울산": "7", "세종": "8", "경기": "31", "강원": "32",
    "충북": "33", "충남": "34", "경북": "35", "경남": "36",
    "전북": "37", "전남": "38", "제주": "39"
}

CONTENT_TYPES = {
    "관광지": "12", "문화시설": "14", "축제": "15", "여행코스": "25",
    "레포츠": "28", "숙박": "32", "쇼핑": "38", "음식점": "39", "카페": "39"
}


# ========== Request/Response 모델 ==========
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7

class MCPQueryRequest(BaseModel):
    query: str  # 자연어 쿼리: "강릉 바다 근처 맛집 추천해줘"
    area_code: Optional[str] = None  # 모바일에서 선택한 도 코드 (예: "32" for 강원)
    sigungu_code: Optional[str] = None  # 모바일에서 선택한 시/군/구 코드
    max_tokens: int = 1024
    temperature: float = 0.3


# ========== LLM 생성 함수 ==========
def generate_response(messages: list[dict], max_tokens: int = 512, temperature: float = 0.7) -> str:
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )


# ========== MCP 도구 호출 ==========
async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """MCP 서버의 도구 호출 (HTTP API)"""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                f"{MCP_SERVER_URL}/api/call_tool",
                json={"name": tool_name, "arguments": arguments}
            )
            result = response.json()
            if result.get("success"):
                return result.get("result", {})
            else:
                return {"error": result.get("error", "Unknown error")}
        except Exception as e:
            return {"error": str(e)}


async def call_mcp_tool_direct(tool_name: str, arguments: dict) -> dict:
    """MCP 서버 직접 호출 (HTTP API)"""
    # tour-mcp-server의 함수를 직접 HTTP로 호출
    # FastMCP는 기본적으로 stdio지만, HTTP wrapper 추가 필요

    # 임시: 직접 Tour API 호출
    async with httpx.AsyncClient(timeout=30) as client:
        base_url = "https://apis.data.go.kr/B551011/KorService2"
        api_key = os.getenv("TOUR_API_KEY", "")

        common_params = {
            "serviceKey": api_key,
            "MobileOS": "ETC",
            "MobileApp": "TravelMCP",
            "_type": "json",
            "numOfRows": arguments.get("num_of_rows", 20)
        }

        endpoint_map = {
            "get_area_codes": "areaCode2",
            "search_by_area": "areaBasedList2",
            "search_by_keyword": "searchKeyword2",
            "search_by_location": "locationBasedList2",
            "search_festivals": "searchFestival2",
            "search_stays": "searchStay2",
            "get_detail_common": "detailCommon2",
            "get_detail_intro": "detailIntro2",
            "get_detail_images": "detailImage2",
        }

        endpoint = endpoint_map.get(tool_name)
        if not endpoint:
            return {"error": f"Unknown tool: {tool_name}"}

        params = {**common_params}

        # 파라미터 매핑
        if "area_code" in arguments:
            params["areaCode"] = arguments["area_code"]
        if "sigungu_code" in arguments:
            params["sigunguCode"] = arguments["sigungu_code"]
        if "content_type_id" in arguments:
            params["contentTypeId"] = arguments["content_type_id"]
        if "keyword" in arguments:
            params["keyword"] = arguments["keyword"]
        if "map_x" in arguments:
            params["mapX"] = arguments["map_x"]
        if "map_y" in arguments:
            params["mapY"] = arguments["map_y"]
        if "radius" in arguments:
            params["radius"] = arguments["radius"]
        if "event_start_date" in arguments:
            params["eventStartDate"] = arguments["event_start_date"]
        if "content_id" in arguments:
            params["contentId"] = arguments["content_id"]
            params["defaultYN"] = "Y"
            params["firstImageYN"] = "Y"
            params["addrinfoYN"] = "Y"
            params["overviewYN"] = "Y"

        try:
            response = await client.get(f"{base_url}/{endpoint}", params=params)
            data = response.json()

            if data["response"]["header"]["resultCode"] != "0000":
                return {"error": data["response"]["header"]["resultMsg"]}

            items = data["response"]["body"].get("items", {})
            if not items:
                return {"items": [], "totalCount": 0}

            item_list = items.get("item", [])
            if isinstance(item_list, dict):
                item_list = [item_list]

            return {
                "items": item_list[:20],  # 최대 20개
                "totalCount": data["response"]["body"].get("totalCount", len(item_list))
            }
        except Exception as e:
            return {"error": str(e)}


# ========== LLM 기반 도구 선택 ==========
def select_tools_with_llm(query: str, area_code: Optional[str] = None, sigungu_code: Optional[str] = None) -> list[dict]:
    """LLM을 사용해 쿼리에 맞는 도구와 파라미터 선택"""

    tools_description = "\n".join([
        f"- {t['name']}: {t['description']}\n  파라미터: {t['parameters']}" for t in MCP_TOOLS
    ])

    # area_code가 제공된 경우 프롬프트에 명시적으로 주입
    area_context = ""
    if area_code:
        area_context = f"""
**🔴 중요: 사용자가 이미 지역을 선택했습니다 🔴**
- area_code: "{area_code}" (이 코드를 반드시 사용하세요. 다른 지역코드를 사용하지 마세요)
"""
        if sigungu_code:
            area_context += f'- sigungu_code: "{sigungu_code}" (이 코드를 반드시 사용하세요)\n'
        area_context += "\n질문에서 지역명을 추출하지 말고, 위에 제공된 area_code를 그대로 사용하세요.\n"

    prompt = f"""당신은 여행 정보 검색을 위한 도구 선택 AI입니다.
사용자의 질문을 분석하고, 적절한 도구와 파라미터를 JSON 형식으로 반환하세요.

{area_context}
## 사용 가능한 도구:
{tools_description}

## 지역코드 (area_code) - 사용자가 제공하지 않은 경우에만 참고:
서울=1, 인천=2, 대전=3, 대구=4, 광주=5, 부산=6, 울산=7, 세종=8
경기=31, 강원=32, 충북=33, 충남=34, 경북=35, 경남=36, 전북=37, 전남=38, 제주=39

## 콘텐츠타입 (content_type_id):
관광지=12, 문화시설=14, 축제=15, 여행코스=25, 레포츠=28, 숙박=32, 쇼핑=38, 음식점=39

## 예시 (area_code + sigungu_code 제공된 경우):
질문: "바다 근처 맛집 추천해줘"
제공된 area_code: "32", sigungu_code: "1"
응답: {{"tools": [{{"name": "search_by_keyword", "arguments": {{"keyword": "바다 맛집", "area_code": "32", "sigungu_code": "1", "content_type_id": "39"}}}}]}}

질문: "맛집 추천해줘"
제공된 area_code: "32", sigungu_code: "1"
응답: {{"tools": [{{"name": "search_by_keyword", "arguments": {{"keyword": "맛집", "area_code": "32", "sigungu_code": "1", "content_type_id": "39"}}}}]}}

## 예시 (area_code만 제공된 경우):
질문: "조용한 관광지 추천"
제공된 area_code: "39"
응답: {{"tools": [{{"name": "search_by_keyword", "arguments": {{"keyword": "조용한 관광지", "area_code": "39", "content_type_id": "12"}}}}]}}

## 예시 (area_code가 제공되지 않은 경우):
질문: "강릉 바다 근처 맛집 추천해줘"
응답: {{"tools": [{{"name": "search_by_keyword", "arguments": {{"keyword": "강릉 맛집", "area_code": "32", "content_type_id": "39", "num_of_rows": 20}}}}]}}

질문: "부산 해운대 근처 숙박과 맛집"
응답: {{"tools": [{{"name": "search_by_keyword", "arguments": {{"keyword": "해운대 숙박", "area_code": "6", "content_type_id": "32"}}}}, {{"name": "search_by_keyword", "arguments": {{"keyword": "해운대 맛집", "area_code": "6", "content_type_id": "39"}}}}]}}

## 중요:
- **area_code가 위에 제공된 경우 반드시 그 값을 사용 (최우선)**
- **sigungu_code가 제공된 경우 반드시 arguments에 포함 (최우선)**
- 제공되지 않은 경우에만 질문에서 지역명을 추출하여 area_code로 변환
- 음식점/맛집/카페는 content_type_id="39"
- 숙박/호텔/펜션은 content_type_id="32"
- 관광지/명소는 content_type_id="12"
- optional 파라미터는 확실한 값이 있을 때만 포함, 없으면 생략
- 키워드 검색(search_by_keyword)이 가장 유연함
- 여러 조건이 있으면 도구를 여러 개 사용
- 모든 값은 실제 데이터만 입력 (설명문 금지)

## 사용자 질문:
{query}

## 응답 (JSON만 출력, 설명 없이):
{{"tools": [...]}}"""

    messages = [{"role": "user", "content": prompt}]
    response = generate_response(messages, max_tokens=500, temperature=0.1)

    # JSON 파싱
    try:
        # JSON 부분만 추출
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            return result.get("tools", [])
    except json.JSONDecodeError:
        pass

    # 파싱 실패시 기본값
    return []


def curate_results_with_llm(query: str, tool_results: list[dict]) -> dict:
    """LLM을 사용해 검색 결과를 큐레이션 - spots(리스트뷰) + course(코스뷰) 분리"""

    # 결과 요약 (좌표 정보 포함)
    results_summary = []
    for result in tool_results:
        if "items" in result and result["items"]:
            for item in result["items"][:15]:
                results_summary.append({
                    "title": item.get("title", ""),
                    "addr": item.get("addr1", ""),
                    "type": item.get("contenttypeid", ""),
                    "image": item.get("firstimage", ""),
                    "mapx": item.get("mapx", ""),  # 경도
                    "mapy": item.get("mapy", ""),  # 위도
                    "tel": item.get("tel", ""),
                    "content_id": item.get("contentid", "")
                })

    if not results_summary:
        return {
            "spots": [],
            "course": None,
            "message": "요청하신 조건에 맞는 장소를 찾지 못했습니다."
        }

    prompt = f"""당신은 코레일 동행열차 여행 큐레이터입니다.

## 서비스 컨텍스트:
- 대상: **커플 여행객** (코레일 동행열차 서비스)
- 목적: 관광/데이트
- 분위기: 로맨틱하고 특별한 추억 만들기

## 사용자 요청:
{query}

## 검색된 장소들 (총 {len(results_summary)}개):
{json.dumps(results_summary, ensure_ascii=False, indent=2)}

## 응답 형식 (JSON만 출력, 설명 없이):
{{
  "course": {{
    "title": "코스 제목 (예: 강릉 바다향 데이트 코스)",
    "stops": [
      {{
        "order": 1,
        "name": "장소명",
        "address": "주소",
        "mapx": "경도값",
        "mapy": "위도값",
        "content_id": "콘텐츠ID",
        "category": "카페/음식점/관광지/숙박",
        "time": "오전 10시",
        "duration": "1시간",
        "reason": "커플에게 추천하는 이유",
        "tip": "방문 팁"
      }}
    ],
    "total_duration": "약 6시간",
    "summary": "코스 요약 (2-3문장, 커플 여행 관점)"
  }}
}}

## 규칙:
- 사용자 요청에 맞게 3~6개 장소를 **동선 순서대로** 선정
- **커플 데이트 관점**에서 추천 이유 작성
- mapx, mapy 값이 있는 장소 우선 선택 (지도 연동용)
- content_id 반드시 포함 (상세정보 조회용)
- 중복/비슷한 장소 제외
- 반드시 유효한 JSON만 출력"""

    messages = [{"role": "user", "content": prompt}]
    response = generate_response(messages, max_tokens=1500, temperature=0.5)

    # JSON 파싱
    curated_course = None
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            curated_course = parsed.get("course")
    except json.JSONDecodeError:
        pass

    # spots 리스트 생성 (전체 검색 결과, 좌표 포함)
    spots = []
    for r in results_summary:
        spots.append({
            "name": r["title"],
            "address": r["addr"],
            "category": _get_category_name(r["type"]),
            "image_url": r["image"],
            "mapx": r["mapx"],
            "mapy": r["mapy"],
            "tel": r["tel"],
            "content_id": r["content_id"]
        })

    return {
        "spots": spots,  # 리스트 뷰용 (전체)
        "course": curated_course,  # 코스 뷰용 (LLM 큐레이션)
        "message": f"{len(spots)}개의 장소를 찾았습니다."
    }


def _get_category_name(content_type_id: str) -> str:
    """content_type_id를 카테고리명으로 변환"""
    type_map = {
        "12": "관광지",
        "14": "문화시설",
        "15": "축제/행사",
        "25": "여행코스",
        "28": "레포츠",
        "32": "숙박",
        "38": "쇼핑",
        "39": "음식점"
    }
    return type_map.get(content_type_id, "기타")


# ========== API 엔드포인트 ==========
@app.get("/health")
async def health():
    return {"status": "ok", "mcp_enabled": True}

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [{"id": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct", "object": "model"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """기존 OpenAI 호환 엔드포인트"""
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    response_text = generate_response(messages, request.max_tokens, request.temperature)

    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }]
    }


@app.post("/v1/mcp/query")
async def mcp_query(request: MCPQueryRequest):
    """
    MCP Host 엔드포인트
    자연어 쿼리 → LLM이 도구 선택 → 도구 실행 → 결과 큐레이션

    응답 구조:
    - spots: 리스트 뷰용 (전체 검색 결과, 좌표 포함)
    - course: 코스 뷰용 (LLM이 큐레이션한 동선)
    """
    query = request.query
    area_code = request.area_code
    sigungu_code = request.sigungu_code

    # 1. LLM으로 도구 선택 (area 정보 전달)
    print(f"[MCP] Query: {query}, area_code: {area_code}, sigungu_code: {sigungu_code}")
    selected_tools = select_tools_with_llm(query, area_code, sigungu_code)
    print(f"[MCP] Selected tools: {selected_tools}")

    if not selected_tools:
        return {
            "success": False,
            "error": "적절한 도구를 찾지 못했습니다.",
            "query": query,
            "spots": [],
            "course": None
        }

    # 2. 선택된 도구들 실행
    tool_results = []
    for tool in selected_tools:
        tool_name = tool.get("name")
        arguments = tool.get("arguments", {})

        print(f"[MCP] Calling tool: {tool_name} with {arguments}")
        result = await call_mcp_tool(tool_name, arguments)
        tool_results.append({
            "tool": tool_name,
            "arguments": arguments,
            "result": result
        })
        print(f"[MCP] Tool result: {len(result.get('items', []))} items")

    # 3. 결과 큐레이션 (spots + course 분리)
    curated = curate_results_with_llm(query, [r["result"] for r in tool_results])

    return {
        "success": True,
        "query": query,
        "area_code": area_code,
        "sigungu_code": sigungu_code,
        "selected_tools": selected_tools,
        "spots": curated.get("spots", []),      # 리스트 뷰용
        "course": curated.get("course"),         # 코스 뷰용
        "message": curated.get("message", "")
    }


@app.get("/v1/mcp/tools")
async def list_mcp_tools():
    """사용 가능한 MCP 도구 목록"""
    return {"tools": MCP_TOOLS}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=30000)
