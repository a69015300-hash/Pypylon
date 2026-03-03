"""
使用 fastapi_mcp 建立 MCP 伺服器
自動將 FastAPI 端點轉換為 MCP 工具

安裝：
    pip install -r requirements.txt

啟動方式：
    uvicorn test_MCP:app --host 0.0.0.0 --port 8000 --reload

端點：
    FastAPI 文件: http://localhost:8000/docs
    MCP 端點: http://localhost:8000/mcp
"""

import base64
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import ezdxf
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query, UploadFile, File
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# fastapi_mcp - 自動將 FastAPI 端點轉換為 MCP 工具
from fastapi_mcp import FastApiMCP


# ══════════════════════════════════════════════════════════════════════════════
# 全域變數
# ══════════════════════════════════════════════════════════════════════════════

# 暫存
image_cache: Dict[str, dict] = {}
contours_cache: Dict[str, dict] = {}
CACHE_EXPIRE_SECONDS = 300


# ══════════════════════════════════════════════════════════════════════════════
# 影像處理函式
# ══════════════════════════════════════════════════════════════════════════════

def threhold(img: np.ndarray, thresh: int, maxval: int = 255, type: int = cv2.THRESH_BINARY) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _, th = cv2.threshold(img, thresh, maxval, type)
    return th


def connectedComponentsOnly(img: np.ndarray, min_area: int, max_area: int):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    filtered_mask = np.zeros_like(img, dtype=np.uint8)
    all_contours = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if not (min_area <= area <= max_area):
            continue
        comp_mask = ((labels == label).astype(np.uint8)) * 255
        cnts_info = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
        if not contours:
            continue
        filled = np.zeros_like(comp_mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)
        filtered_mask = cv2.bitwise_or(filtered_mask, filled)
        all_contours.extend(contours)
    return filtered_mask, all_contours


def erode_image(img: np.ndarray, kernel_size: int = 11, iterations: int = 1) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(img, kernel, iterations=iterations)


def calculate_contours_center(contours, img_shape, scale: float = 1.0) -> Tuple[float, float]:
    H = img_shape[0]
    all_pts = []
    for cnt in contours:
        if cnt is not None and len(cnt) >= 2:
            pts = cnt.reshape(-1, 2)
            all_pts.extend(pts)
    if len(all_pts) == 0:
        return (0.0, 0.0)
    all_pts = np.array(all_pts)
    cx_img = np.mean(all_pts[:, 0])
    cy_img = np.mean(all_pts[:, 1])
    return (cx_img * scale, (H - 1 - cy_img) * scale)


def contours_to_dxf_as_block(
    contours, img_shape, dxf_path: str = "output.dxf", scale: float = 1.0,
    eps_ratio: float = 0.002, block_name: str = "PART_CONTOURS", layer_name: str = "CONTOUR",
    reference_center: Optional[Tuple[float, float]] = None
) -> str:
    H = img_shape[0]
    if reference_center is not None:
        cx_dxf, cy_dxf = reference_center
    else:
        cx_dxf, cy_dxf = calculate_contours_center(contours, img_shape, scale)
    doc = ezdxf.new("R2007")
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    if layer_name not in doc.layers:
        doc.layers.new(name=layer_name)
    block = doc.blocks.new(name=block_name)
    for cnt in contours:
        if cnt is None or len(cnt) < 2:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = max(1e-6, eps_ratio * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 2:
            continue
        dxf_pts = [(float(x) * scale - cx_dxf, float(H - 1 - y) * scale - cy_dxf) for x, y in pts]
        block.add_lwpolyline(dxf_pts, format="xy", close=True, dxfattribs={"layer": layer_name})
    msp.add_blockref(block_name, insert=(0, 0, 0), dxfattribs={'xscale': 1.0, 'yscale': 1.0, 'rotation': 0})
    doc.update_extents()
    doc.audit()
    doc.saveas(dxf_path)
    return dxf_path


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def contours_to_json(contours) -> list:
    return [cnt.reshape(-1, 2).tolist() for cnt in contours]

def json_to_contours(data: list) -> list:
    return [np.array(pts, dtype=np.int32).reshape(-1, 1, 2) for pts in data]

def cleanup_expired_cache():
    now = time.time()
    for cache in [image_cache, contours_cache]:
        expired = [k for k, v in cache.items() if now - v["timestamp"] > CACHE_EXPIRE_SECONDS]
        for k in expired:
            del cache[k]

def cache_image(img_bytes: bytes) -> str:
    cleanup_expired_cache()
    image_id = str(uuid.uuid4())
    image_cache[image_id] = {"data": img_bytes, "timestamp": time.time()}
    return image_id

def cache_contours(contours: list, img_shape: tuple) -> str:
    cleanup_expired_cache()
    contours_id = str(uuid.uuid4())
    contours_cache[contours_id] = {"data": contours_to_json(contours), "img_shape": img_shape, "timestamp": time.time()}
    return contours_id


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 應用
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="影像處理 MCP API",
    description="使用 fastapi_mcp 自動將 FastAPI 端點轉換為 MCP 工具 - 影像處理、輪廓分析、DXF 匯出",
    version="1.0.0",
)

# CORS 設定 - 允許外部存取
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 靜態檔案與首頁 ────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, tags=["Web"])
def serve_index():
    """提供網頁介面"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return HTMLResponse("<h1>影像處理 MCP API</h1><p>請訪問 /docs 查看 API 文件</p>")


# ── 健康檢查 ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="健康檢查")
def health_check():
    """系統健康檢查端點"""
    return {"status": "ok", "message": "服務運行中"}


# ── 圖片上傳端點 ──────────────────────────────────────────────────────────────

@app.post("/image/upload", tags=["Image"], summary="上傳圖片")
async def upload_image(request: Request, file: UploadFile = File(...)):
    """上傳圖片檔案，回傳圖片 URL 和 ID"""
    contents = await file.read()

    # 驗證是否為有效圖片
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="無法解碼圖片，請確認檔案格式")

    # 統一轉為 BMP 暫存
    ok, buf = cv2.imencode(".bmp", img)
    if not ok:
        raise HTTPException(status_code=500, detail="圖片編碼失敗")

    image_id = cache_image(buf.tobytes())
    base_url = str(request.base_url).rstrip("/")
    return {
        "image_url": f"{base_url}/image/{image_id}",
        "image_id": image_id,
        "filename": file.filename,
        "width": img.shape[1],
        "height": img.shape[0],
    }


# ── 圖片存取端點 ──────────────────────────────────────────────────────────────

@app.get("/image/{image_id}", tags=["Image"], summary="取得暫存圖片")
def get_image(image_id: str):
    """取得暫存的圖片"""
    if image_id not in image_cache:
        raise HTTPException(status_code=404, detail="圖片不存在或已過期")
    return Response(content=image_cache[image_id]["data"], media_type="image/bmp")


# ── 影像處理端點 ──────────────────────────────────────────────────────────────

@app.post("/image/threshold", tags=["Image Processing"], summary="二值化處理")
def image_threshold(
    request: Request,
    image_id: str = Query(..., description="圖片 ID"),
    thresh: int = Query(default=100, description="閾值 (0-255)"),
    maxval: int = Query(default=255, description="最大值"),
    type: int = Query(default=0, description="閾值類型: 0=BINARY, 1=BINARY_INV")
):
    """對圖片進行二值化處理"""
    if image_id not in image_cache:
        raise HTTPException(status_code=404, detail="圖片不存在或已過期")
    arr = np.frombuffer(image_cache[image_id]["data"], dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="無法解碼圖片")
    result = threhold(img, thresh, maxval, type)
    ok, buf = cv2.imencode(".bmp", result)
    new_image_id = cache_image(buf.tobytes())
    base_url = str(request.base_url).rstrip("/")
    return {"image_url": f"{base_url}/image/{new_image_id}", "image_id": new_image_id}


@app.post("/image/connected-components", tags=["Image Processing"], summary="連通元件分析")
def image_connected_components(
    request: Request,
    image_id: str = Query(..., description="圖片 ID"),
    min_area: int = Query(default=20000, description="最小面積 (像素)"),
    max_area: int = Query(default=300000, description="最大面積 (像素)"),
    draw_contours: bool = Query(default=False, description="繪製輪廓線")
):
    """連通元件分析，篩選指定面積範圍的物件"""
    if image_id not in image_cache:
        raise HTTPException(status_code=404, detail="圖片不存在或已過期")
    arr = np.frombuffer(image_cache[image_id]["data"], dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="無法解碼圖片")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_mask, contours = connectedComponentsOnly(img, min_area, max_area)
    if draw_contours and len(contours) > 0:
        output_img = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2)
    else:
        output_img = filtered_mask
    ok, buf = cv2.imencode(".bmp", output_img)
    new_image_id = cache_image(buf.tobytes())
    contours_id = cache_contours(contours, img.shape)
    base_url = str(request.base_url).rstrip("/")
    return {
        "image_url": f"{base_url}/image/{new_image_id}",
        "image_id": new_image_id,
        "contours_id": contours_id,
        "contour_count": len(contours)
    }


@app.post("/image/erode", tags=["Image Processing"], summary="形態學侵蝕")
def image_erode(
    request: Request,
    image_id: str = Query(..., description="圖片 ID"),
    kernel_size: int = Query(default=11, description="侵蝕核心大小"),
    iterations: int = Query(default=1, description="侵蝕次數")
):
    """形態學侵蝕，縮小白色區域"""
    if image_id not in image_cache:
        raise HTTPException(status_code=404, detail="圖片不存在或已過期")
    arr = np.frombuffer(image_cache[image_id]["data"], dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="無法解碼圖片")
    eroded = erode_image(img, kernel_size, iterations)
    ok, buf = cv2.imencode(".bmp", eroded)
    new_image_id = cache_image(buf.tobytes())
    base_url = str(request.base_url).rstrip("/")
    return {"image_url": f"{base_url}/image/{new_image_id}", "image_id": new_image_id}


@app.post("/image/find-centers", tags=["Image Processing"], summary="抓取物體中心座標")
def find_centers(
    contours_id: str = Query(..., description="輪廓 ID"),
    scale: float = Query(default=0.00274, description="縮放比例 (像素轉毫米)")
):
    """抓取物體中心位置座標，自動判斷形狀"""
    if contours_id not in contours_cache:
        raise HTTPException(status_code=404, detail="輪廓資料不存在或已過期")
    contours = json_to_contours(contours_cache[contours_id]["data"])
    centers = []
    for i, cnt in enumerate(contours):
        if len(cnt) < 5:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        area = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1] if rect[1][0] > 0 and rect[1][1] > 0 else 0
        rect_ratio = area / rect_area if rect_area > 0 else 0
        (_, _), radius = cv2.minEnclosingCircle(cnt)
        circle_area = 3.14159 * radius * radius
        circle_ratio = area / circle_area if circle_area > 0 else 0
        shape = "rectangle" if rect_ratio > 0.85 else ("circle" if circle_ratio > 0.75 else "irregular")
        centers.append({
            "index": i, "shape": shape,
            "center_pixel": {"x": cx, "y": cy},
            "center_mm": {"x": round(cx * scale, 4), "y": round(cy * scale, 4)},
            "area_pixel": int(area)
        })
    return {"centers": centers, "count": len(centers), "scale": scale, "unit": "mm"}


# ── DXF 匯出端點 ──────────────────────────────────────────────────────────────

@app.post("/export/dxf", tags=["Export"], summary="匯出 DXF")
def export_dxf(
    contours_id: str = Query(..., description="輪廓 ID"),
    scale: float = Query(default=0.00274, description="縮放比例"),
    eps_ratio: float = Query(default=0.002, description="Douglas-Peucker 簡化比例"),
    filename: str = Query(default="output.dxf", description="輸出檔名"),
    block_name: str = Query(default="PART_CONTOURS", description="DXF BLOCK 名稱")
):
    """將輪廓轉為 DXF 檔案下載"""
    if contours_id not in contours_cache:
        raise HTTPException(status_code=404, detail="輪廓資料不存在或已過期")
    cache_data = contours_cache[contours_id]
    contours = json_to_contours(cache_data["data"])
    img_shape = cache_data["img_shape"]
    tmp_dir = tempfile.mkdtemp()
    dxf_path = os.path.join(tmp_dir, filename)
    contours_to_dxf_as_block(contours=contours, img_shape=img_shape, dxf_path=dxf_path, scale=scale, eps_ratio=eps_ratio, block_name=block_name)
    return FileResponse(path=dxf_path, filename=filename, media_type="application/dxf")


# ══════════════════════════════════════════════════════════════════════════════
# 建立 MCP 伺服器 (使用 fastapi_mcp)
# ══════════════════════════════════════════════════════════════════════════════

mcp = FastApiMCP(
    app,
    name="vision-mcp",
    description="影像處理 MCP - 圖片上傳、影像處理、輪廓分析、DXF 匯出",
)

mcp.mount_http()


# ══════════════════════════════════════════════════════════════════════════════
# 啟動
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="影像處理 MCP Server (fastapi_mcp)")
    parser.add_argument("--host", default="0.0.0.0", help="主機位址")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="埠號")

    args = parser.parse_args()

    print("=" * 60)
    print("影像處理 MCP Server (fastapi_mcp)")
    print("=" * 60)
    print(f"FastAPI 文件: http://{args.host}:{args.port}/docs")
    print(f"MCP 端點: http://{args.host}:{args.port}/mcp")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port)
