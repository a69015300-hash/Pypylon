"""
FastAPI + MCP for  影像處理系統
同時支援 REST API 和 MCP (Model Context Protocol)

啟動方式：
    # 同時運行 FastAPI (8000) + MCP (8001)
    python api.py

    # 僅 FastAPI
    python api.py --mode fastapi

    # 僅 MCP (Streamable HTTP)
    python api.py --mode mcp
"""

import argparse
import base64
import os
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import ezdxf
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field
from pypylon import pylon

# MCP imports
from mcp.server.fastmcp import FastMCP


# ══════════════════════════════════════════════════════════════════════════════
# BaslerCamera 類別
# ══════════════════════════════════════════════════════════════════════════════

class BaslerCamera:
    """
    將 pypylon 相機操作包成可重複使用的物件：
    - open() / close()
    - grab_one_bgr()：抓一張 BGR8 圖（numpy array）
    - grab_loop()：連續抓圖並把影像丟給 callback 做後續影像處理
    """

    def __init__(self, ip: Optional[str] = None):
        self.ip = ip
        self.camera: Optional[pylon.InstantCamera] = None

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def _pick_device(self):
        tl = pylon.TlFactory.GetInstance()
        devs = tl.EnumerateDevices()
        if not devs:
            raise RuntimeError("No camera found")

        if self.ip:
            for d in devs:
                if hasattr(d, "GetIpAddress") and d.GetIpAddress() == self.ip:
                    return tl.CreateDevice(d)

        return tl.CreateDevice(devs[0])

    @staticmethod
    def _safe_set(fn):
        try:
            fn()
            return True
        except Exception:
            return False

    def open(self):
        if self.camera and self.camera.IsOpen():
            return

        self.camera = pylon.InstantCamera(self._pick_device())
        self.camera.Open()

        self._safe_set(lambda: self.camera.TriggerSelector.SetValue("FrameStart"))
        self._safe_set(lambda: self.camera.TriggerMode.SetValue("Off"))
        self._safe_set(lambda: self.camera.GevSCPSPacketSize.SetValue(1500))
        self._safe_set(lambda: self.camera.GevSCPD.SetValue(0))

        try:
            self.camera.MaxNumBuffer = 100
        except Exception:
            pass

    def close(self):
        if not self.camera:
            return
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        except Exception:
            pass
        try:
            if self.camera.IsOpen():
                self.camera.Close()
        except Exception:
            pass
        self.camera = None

    def grab_one_bgr(self, timeout_ms: int = 5000, retries: int = 3):
        """抓一張 BGR 影像（numpy array）。失敗會重試 retries 次。"""
        if not self.camera or not self.camera.IsOpen():
            self.open()

        assert self.camera is not None

        last_err = None
        for _ in range(retries):
            try:
                self.camera.StartGrabbingMax(1)
                grab = self.camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
                try:
                    if grab.GrabSucceeded():
                        return self.converter.Convert(grab).GetArray()
                    last_err = (grab.ErrorCode, grab.ErrorDescription)
                finally:
                    grab.Release()
                    self.camera.StopGrabbing()
            except (pylon.TimeoutException, pylon.GenericException) as e:
                last_err = str(e)
                time.sleep(0.05)

        raise RuntimeError(f"Grab failed: {last_err}")

    def grab_loop(
        self,
        on_image: Callable,
        stop_key: int = 27,
        window_name: Optional[str] = None,
        timeout_ms: int = 5000
    ):
        """連續抓圖，把每張 img 丟給 on_image(img) 做後續處理。"""
        if not self.camera or not self.camera.IsOpen():
            self.open()
        assert self.camera is not None

        if window_name:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        try:
            while self.camera.IsGrabbing():
                try:
                    grab = self.camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
                    try:
                        if grab.GrabSucceeded():
                            img = self.converter.Convert(grab).GetArray()
                            on_image(img)
                            if window_name:
                                cv2.imshow(window_name, img)
                                if (cv2.waitKey(1) & 0xFF) == stop_key:
                                    break
                    finally:
                        grab.Release()
                except pylon.TimeoutException:
                    continue
                except pylon.GenericException:
                    continue
        finally:
            try:
                self.camera.StopGrabbing()
            except Exception:
                pass
            if window_name:
                cv2.destroyWindow(window_name)


# ══════════════════════════════════════════════════════════════════════════════
# 共用影像處理函式
# ══════════════════════════════════════════════════════════════════════════════

def threhold(img: np.ndarray, thresh: int, maxval: int = 255, type: int = cv2.THRESH_BINARY) -> np.ndarray:
    """
    二值化處理，參數與 cv2.threshold 一致

    Args:
        img: 輸入影像
        thresh: 閾值
        maxval: 超過閾值時的最大值
        type: 閾值類型
            0 = THRESH_BINARY
            1 = THRESH_BINARY_INV
            2 = THRESH_TRUNC
            3 = THRESH_TOZERO
            4 = THRESH_TOZERO_INV
            8 = THRESH_OTSU (可與上述組合，如 0+8=8, 1+8=9)
            16 = THRESH_TRIANGLE (可與上述組合)
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _, th = cv2.threshold(img, thresh, maxval, type)
    return th


def connectedComponentsOnly(img: np.ndarray, min_area: int, max_area: int):
    """
    純連通元件分析（不含侵蝕）：篩選面積 + 提取輪廓。
    回傳 (filtered_mask, contours)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )

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


def find_circle(c: np.ndarray) -> Tuple[Tuple[int, int], int]:
    """找最小外接圓，回傳 (center, radius)"""
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


def erode_image(img: np.ndarray, kernel_size: int = 11, iterations: int = 1) -> np.ndarray:
    """形態學侵蝕"""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(img, kernel, iterations=iterations)


def calculate_contours_center(contours, img_shape, scale: float = 1.0) -> Tuple[float, float]:
    """計算所有輪廓的整體質心 (DXF 座標)"""
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
    cx_dxf = cx_img * scale
    cy_dxf = (H - 1 - cy_img) * scale
    return (cx_dxf, cy_dxf)


def contours_to_dxf_as_block(
    contours,
    img_shape,
    dxf_path: str = "output_block.dxf",
    scale: float = 1.0,
    eps_ratio: float = 0.002,
    block_name: str = "PART_CONTOURS",
    layer_name: str = "CONTOUR",
    reference_center: Optional[Tuple[float, float]] = None,
) -> str:
    """
    將所有輪廓定義為單一 BLOCK，RoboDK 會視為單一物件

    Args:
        reference_center: 參考原點 (cx_dxf, cy_dxf)，若為 None 則自動計算
    """
    H = img_shape[0]

    # 計算或使用指定的參考原點
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

        dxf_pts = [
            (
                float(x) * scale - cx_dxf,
                float(H - 1 - y) * scale - cy_dxf
            )
            for x, y in pts
        ]

        block.add_lwpolyline(
            dxf_pts,
            format="xy",
            close=True,
            dxfattribs={"layer": layer_name},
        )

    msp.add_blockref(
        block_name,
        insert=(0, 0, 0),
        dxfattribs={
            'xscale': 1.0,
            'yscale': 1.0,
            'rotation': 0
        }
    )

    doc.update_extents()
    doc.audit()
    doc.saveas(dxf_path)

    return dxf_path


# ══════════════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════════════

def img_to_base64(img: np.ndarray) -> str:
    """numpy array → base64 PNG string"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def base64_to_img(b64: str) -> np.ndarray:
    """base64 string → numpy array"""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("無法解碼 base64 影像")
    return img


def contours_to_json(contours) -> list:
    """OpenCV contours → JSON serializable list"""
    return [cnt.reshape(-1, 2).tolist() for cnt in contours]


def json_to_contours(data: list) -> list:
    """JSON list → OpenCV contour format (list of numpy arrays)"""
    return [np.array(pts, dtype=np.int32).reshape(-1, 1, 2) for pts in data]


def read_image_file(file_path: str) -> np.ndarray:
    """讀取圖片檔案"""
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"無法讀取圖片: {file_path}")
    return img


def save_image_file(img: np.ndarray, file_path: str) -> str:
    """儲存圖片檔案"""
    cv2.imwrite(file_path, img)
    return file_path


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI 設定
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="影像處理 API",
    description="相機操作與影像處理 RESTful API (同時支援 MCP)",
    version="2.0.0",
)

# ── 靜態檔案與首頁 ───────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """提供網頁介面"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return HTMLResponse("<h1> API</h1><p>請訪問 /docs 查看 API 文件</p>")


# ── 暫存機制 ─────────────────────────────────────────────────────────────────
# 內存暫存圖片 {image_id: {"data": bytes, "timestamp": float}}
image_cache: Dict[str, dict] = {}
# 內存暫存輪廓 {contours_id: {"data": list, "img_shape": tuple, "timestamp": float}}
contours_cache: Dict[str, dict] = {}
CACHE_EXPIRE_SECONDS = 300  # 5 分鐘過期

# ── 9點校正資料 ─────────────────────────────────────────────────────────────────
# 儲存校正點位 {"points": [...], "matrix": ndarray or None, "calibrated": bool}
calibration_data = {
    "points": [],           # 校正點位列表 [{"pixel": (x, y), "robot": (x, y)}, ...]
    "matrix": None,         # 仿射變換矩陣 (2x3)
    "calibrated": False,    # 是否已完成校正
    "error": None,          # 校正誤差 (RMSE)
}


# ── Pydantic Models ──────────────────────────────────────────────────────────

class CameraConnectRequest(BaseModel):
    ip: Optional[str] = None


class CameraConnectResponse(BaseModel):
    success: bool
    message: str


class CameraCaptureRequest(BaseModel):
    timeout_ms: int = 5000
    retries: int = 3


class ImageResponse(BaseModel):
    image: str


class ThresholdRequest(BaseModel):
    image: str
    value: int = 100


class ConnectedComponentsRequest(BaseModel):
    image: str
    min_area: int = 20000
    max_area: int = 300000


class ConnectedComponentsResponse(BaseModel):
    image: str
    contours: list


class ErodeRequest(BaseModel):
    image: str
    kernel_size: int = 11
    iterations: int = 1


class FindCircleRequest(BaseModel):
    contours: list


class CircleInfo(BaseModel):
    center: List[float]
    radius: float


class FindCircleResponse(BaseModel):
    circles: List[CircleInfo]


class ExportDxfRequest(BaseModel):
    contours: list
    img_height: int
    img_width: int
    scale: float = 0.15
    eps_ratio: float = 0.002
    dxf_path: str = "output.dxf"
    block_name: str = "PART_CONTOURS"
    layer_name: str = "CONTOUR"


# ── FastAPI 端點 ─────────────────────────────────────────────────────────────

def cleanup_expired_cache():
    """清理過期的暫存資料"""
    now = time.time()
    # 清理圖片暫存
    expired = [k for k, v in image_cache.items() if now - v["timestamp"] > CACHE_EXPIRE_SECONDS]
    for k in expired:
        del image_cache[k]
    # 清理輪廓暫存
    expired = [k for k, v in contours_cache.items() if now - v["timestamp"] > CACHE_EXPIRE_SECONDS]
    for k in expired:
        del contours_cache[k]


def cache_image(img_bytes: bytes) -> str:
    """暫存圖片並回傳 image_id"""
    cleanup_expired_cache()
    image_id = str(uuid.uuid4())
    image_cache[image_id] = {"data": img_bytes, "timestamp": time.time()}
    return image_id


def cache_contours(contours: list, img_shape: tuple) -> str:
    """暫存輪廓並回傳 contours_id"""
    cleanup_expired_cache()
    contours_id = str(uuid.uuid4())
    contours_cache[contours_id] = {
        "data": contours_to_json(contours),
        "img_shape": img_shape,
        "timestamp": time.time()
    }
    return contours_id


@app.get("/image/{image_id}")
def api_get_image(image_id: str):
    """取得暫存的圖片"""
    if image_id not in image_cache:
        raise HTTPException(status_code=404, detail="圖片不存在或已過期")
    return Response(content=image_cache[image_id]["data"], media_type="image/bmp")


@app.post("/camera/connect")
def api_camera_connect(
    ip: Optional[str] = Query(default=None, description="相機 IP 位址 (可選，不填則連接第一台)")
):
    """測試相機連線是否正常"""
    cam = BaslerCamera(ip=ip)
    try:
        cam.open()
        return {"success": True, "message": "連線成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"連線失敗: {e}")
    finally:
        cam.close()


@app.post("/camera/capture")
def api_camera_capture(
    request: Request,
    timeout_ms: int = Query(default=5000, description="超時時間 (毫秒)"),
    retries: int = Query(default=3, description="重試次數")
):
    """拍一張照片，暫存並回傳圖片 URL"""
    cam = BaslerCamera()
    try:
        cam.open()
        img = cam.grab_one_bgr(timeout_ms=timeout_ms, retries=retries)
        ok, buf = cv2.imencode(".bmp", img)
        if not ok:
            raise HTTPException(status_code=500, detail="BMP 編碼失敗")
        image_id = cache_image(buf.tobytes())
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/image/{image_id}"
        return {"image_url": image_url, "image_id": image_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"拍照失敗: {e}")
    finally:
        cam.close()


@app.post("/image/threshold")
def api_image_threshold(
    request: Request,
    image_id: str = Query(..., description="圖片 ID (從 /camera/capture 取得)"),
    thresh: int = Query(default=100, description="閾值 (0-255)"),
    maxval: int = Query(default=255, description="超過閾值時的最大值 (0-255)"),
    type: int = Query(default=0, description="閾值類型: 0=BINARY, 1=BINARY_INV, 2=TRUNC, 3=TOZERO, 4=TOZERO_INV, 8=OTSU, 16=TRIANGLE")
):
    """二值化處理參數，回傳處理後圖片 URL"""
    try:
        if image_id not in image_cache:
            raise HTTPException(status_code=404, detail="圖片不存在或已過期")

        img_bytes = image_cache[image_id]["data"]
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail="無法解碼圖片")

        result = threhold(img, thresh, maxval, type)
        ok, buf = cv2.imencode(".bmp", result)
        if not ok:
            raise HTTPException(status_code=500, detail="BMP 編碼失敗")

        new_image_id = cache_image(buf.tobytes())
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/image/{new_image_id}"
        return {"image_url": image_url, "image_id": new_image_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"二值化失敗: {e}")


@app.post("/image/connected-components")
def api_image_connected_components(
    request: Request,
    image_id: str = Query(..., description="圖片 ID (從 /camera/capture 或 /image/threshold 取得)"),
    min_area: int = Query(default=20000, description="最小面積 (像素)"),
    max_area: int = Query(default=300000, description="最大面積 (像素)"),
    draw_contours: bool = Query(default=False, description="是否在輸出圖片上繪製輪廓線"),
    contour_thickness: int = Query(default=2, description="輪廓線粗細 (像素)")
):
    """連通元件分析（篩選面積），回傳處理後圖片 URL 及輪廓 ID"""
    try:
        if image_id not in image_cache:
            raise HTTPException(status_code=404, detail="圖片不存在或已過期")

        img_bytes = image_cache[image_id]["data"]
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail="無法解碼圖片")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        filtered_mask, contours = connectedComponentsOnly(img, min_area, max_area)

        # 決定輸出圖片：是否繪製輪廓線
        if draw_contours and len(contours) > 0:
            # 驗證輪廓線粗細，錯誤則使用預設值 2
            thickness = contour_thickness if isinstance(contour_thickness, int) and contour_thickness >= 1 else 2
            # 轉為彩色圖以便繪製彩色輪廓
            output_img = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(output_img, contours, -1, (0, 255, 0), thickness)  # 綠色輪廓線
        else:
            output_img = filtered_mask

        ok, buf = cv2.imencode(".bmp", output_img)
        if not ok:
            raise HTTPException(status_code=500, detail="BMP 編碼失敗")

        # 計算每個輪廓的位置資訊
        contours_info = []
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            contours_info.append({
                "index": i,
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "centroid": {"x": cx, "y": cy},
                "area": int(area)
            })

        new_image_id = cache_image(buf.tobytes())
        contours_id = cache_contours(contours, img.shape)
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/image/{new_image_id}"
        return {
            "image_url": image_url,
            "image_id": new_image_id,
            "contours_id": contours_id,
            "contour_count": len(contours),
            "contours_info": contours_info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"連通元件分析失敗: {e}")


@app.post("/image/erode")
def api_image_erode(
    request: Request,
    image_id: str = Query(..., description="圖片 ID"),
    kernel_size: int = Query(default=11, description="侵蝕核心大小 (奇數)"),
    iterations: int = Query(default=1, description="侵蝕次數")
):
    """形態學侵蝕，回傳處理後圖片 URL"""
    try:
        if image_id not in image_cache:
            raise HTTPException(status_code=404, detail="圖片不存在或已過期")

        img_bytes = image_cache[image_id]["data"]
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail="無法解碼圖片")

        eroded = erode_image(img, kernel_size, iterations)
        ok, buf = cv2.imencode(".bmp", eroded)
        if not ok:
            raise HTTPException(status_code=500, detail="BMP 編碼失敗")

        new_image_id = cache_image(buf.tobytes())
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/image/{new_image_id}"
        return {"image_url": image_url, "image_id": new_image_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"侵蝕失敗: {e}")


@app.post("/image/find-circle")
def api_image_find_circle(
    contours_id: str = Query(..., description="輪廓 ID (從 /image/connected-components 取得)")
):
    """找最小外接圓"""
    try:
        if contours_id not in contours_cache:
            raise HTTPException(status_code=404, detail="輪廓資料不存在或已過期")

        contours_data = contours_cache[contours_id]["data"]
        contours = json_to_contours(contours_data)
        circles = []
        for i, cnt in enumerate(contours):
            if len(cnt) < 3:
                continue
            center, radius = find_circle(cnt)
            circles.append({
                "index": i,
                "center": [float(center[0]), float(center[1])],
                "radius": float(radius)
            })
        return {"circles": circles, "count": len(circles)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"找圓失敗: {e}")


@app.post("/image/find-centers")
def api_find_centers(
    contours_id: str = Query(..., description="輪廓 ID (從 /image/connected-components 取得)"),
    scale: float = Query(default=0.00274, description="縮放比例 (像素轉毫米，預設 1px=2.74μm)")
):
    """
    抓取物體中心位置座標，自動判斷形狀（矩形或圓形）
    回傳像素座標及換算後的實際座標 (mm)
    """
    try:
        if contours_id not in contours_cache:
            raise HTTPException(status_code=404, detail="輪廓資料不存在或已過期")

        cache_data = contours_cache[contours_id]
        contours = json_to_contours(cache_data["data"])

        centers = []
        for i, cnt in enumerate(contours):
            if len(cnt) < 5:
                continue

            # 計算質心
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 計算面積
            area = cv2.contourArea(cnt)

            # 最小外接矩形
            rect = cv2.minAreaRect(cnt)
            rect_width, rect_height = rect[1]
            rect_area = rect_width * rect_height if rect_width > 0 and rect_height > 0 else 0
            rect_ratio = area / rect_area if rect_area > 0 else 0

            # 最小外接圓
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(cnt)
            circle_area = 3.14159 * radius * radius
            circle_ratio = area / circle_area if circle_area > 0 else 0

            # 判斷形狀：比較填充率
            # 矩形填充率接近 1，圓形填充率接近 π/4 ≈ 0.785
            if rect_ratio > 0.85:
                shape = "rectangle"
            elif circle_ratio > 0.75:
                shape = "circle"
            else:
                shape = "irregular"

            # 計算實際座標 (mm)
            cx_mm = cx * scale
            cy_mm = cy * scale

            centers.append({
                "index": i,
                "shape": shape,
                "center_pixel": {"x": cx, "y": cy},
                "center_mm": {"x": round(cx_mm, 4), "y": round(cy_mm, 4)},
                "area_pixel": int(area),
                "area_mm2": round(area * scale * scale, 6),
                "bounding_rect": {
                    "width": round(rect_width, 2),
                    "height": round(rect_height, 2),
                    "angle": round(rect[2], 2)
                },
                "enclosing_circle_radius": round(radius, 2)
            })

        return {
            "centers": centers,
            "count": len(centers),
            "scale": scale,
            "unit": "mm"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"抓取中心位置失敗: {e}")


@app.post("/export/dxf")
def api_export_dxf(
    contours_id: str = Query(..., description="輪廓 ID (從 /image/connected-components 取得)"),
    scale: float = Query(default=0.00274, description="縮放比例 (像素轉毫米，預設 1px=2.74μm)"),
    eps_ratio: float = Query(default=0.002, description="Douglas-Peucker 簡化比例"),
    filename: str = Query(default="output.dxf", description="輸出檔名"),
    block_name: str = Query(default="PART_CONTOURS", description="DXF BLOCK 名稱"),
    layer_name: str = Query(default="CONTOUR", description="DXF 圖層名稱"),
    separate_files: bool = Query(default=False, description="True=每個輪廓輸出獨立 DXF (ZIP 打包), False=所有輪廓合併為單一 DXF")
):
    """將輪廓轉為 DXF 檔案下載"""
    try:
        if contours_id not in contours_cache:
            raise HTTPException(status_code=404, detail="輪廓資料不存在或已過期")

        cache_data = contours_cache[contours_id]
        contours = json_to_contours(cache_data["data"])
        img_shape = cache_data["img_shape"]
        tmp_dir = tempfile.mkdtemp()

        if separate_files and len(contours) > 1:
            # 每個輪廓輸出獨立 DXF，打包成 ZIP
            # 先計算所有輪廓的整體質心作為共同原點
            reference_center = calculate_contours_center(contours, img_shape, scale)

            base_name = filename.replace(".dxf", "")
            zip_path = os.path.join(tmp_dir, f"{base_name}.zip")

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for i, cnt in enumerate(contours):
                    single_dxf_name = f"{base_name}_{i+1}.dxf"
                    single_dxf_path = os.path.join(tmp_dir, single_dxf_name)
                    contours_to_dxf_as_block(
                        contours=[cnt],
                        img_shape=img_shape,
                        dxf_path=single_dxf_path,
                        scale=scale,
                        eps_ratio=eps_ratio,
                        block_name=f"{block_name}_{i+1}",
                        layer_name=layer_name,
                        reference_center=reference_center,
                    )
                    zipf.write(single_dxf_path, single_dxf_name)

            return FileResponse(
                path=zip_path,
                filename=f"{base_name}.zip",
                media_type="application/zip",
            )
        else:
            # 所有輪廓合併為單一 DXF
            dxf_path = os.path.join(tmp_dir, filename)
            contours_to_dxf_as_block(
                contours=contours,
                img_shape=img_shape,
                dxf_path=dxf_path,
                scale=scale,
                eps_ratio=eps_ratio,
                block_name=block_name,
                layer_name=layer_name,
            )

            return FileResponse(
                path=dxf_path,
                filename=filename,
                media_type="application/dxf",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DXF 匯出失敗: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 9 點校正 API
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/calibration/status")
def api_calibration_status():
    """
    取得目前校正狀態
    """
    return {
        "point_count": len(calibration_data["points"]),
        "points": calibration_data["points"],
        "calibrated": calibration_data["calibrated"],
        "error": calibration_data["error"],
        "matrix": calibration_data["matrix"].tolist() if calibration_data["matrix"] is not None else None
    }


@app.post("/calibration/add-point")
def api_calibration_add_point(
    pixel_x: float = Query(..., description="相機像素座標 X"),
    pixel_y: float = Query(..., description="相機像素座標 Y"),
    robot_x: float = Query(..., description="手臂座標 X (mm)"),
    robot_y: float = Query(..., description="手臂座標 Y (mm)")
):
    """
    新增一個校正點位
    需要同時提供相機像素座標和手臂實際座標
    """
    if len(calibration_data["points"]) >= 9:
        raise HTTPException(status_code=400, detail="已達到 9 個點位上限，請先重置或刪除點位")

    point = {
        "index": len(calibration_data["points"]) + 1,
        "pixel": {"x": pixel_x, "y": pixel_y},
        "robot": {"x": robot_x, "y": robot_y}
    }
    calibration_data["points"].append(point)

    # 新增點位後，重置校正狀態
    calibration_data["calibrated"] = False
    calibration_data["matrix"] = None
    calibration_data["error"] = None

    return {
        "message": f"已新增第 {point['index']} 個校正點",
        "point": point,
        "point_count": len(calibration_data["points"]),
        "remaining": 9 - len(calibration_data["points"])
    }


@app.post("/calibration/add-point-from-contour")
def api_calibration_add_point_from_contour(
    contours_id: str = Query(..., description="輪廓 ID"),
    contour_index: int = Query(default=0, description="輪廓索引 (預設第一個)"),
    robot_x: float = Query(..., description="手臂座標 X (mm)"),
    robot_y: float = Query(..., description="手臂座標 Y (mm)")
):
    """
    從輪廓自動抓取中心點作為像素座標，並搭配手臂座標新增校正點
    """
    if contours_id not in contours_cache:
        raise HTTPException(status_code=404, detail="輪廓資料不存在或已過期")

    if len(calibration_data["points"]) >= 9:
        raise HTTPException(status_code=400, detail="已達到 9 個點位上限，請先重置")

    cache_data = contours_cache[contours_id]
    contours = json_to_contours(cache_data["data"])

    if contour_index >= len(contours):
        raise HTTPException(status_code=400, detail=f"輪廓索引超出範圍 (共 {len(contours)} 個)")

    cnt = contours[contour_index]
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        raise HTTPException(status_code=400, detail="無法計算輪廓中心點")

    pixel_x = M["m10"] / M["m00"]
    pixel_y = M["m01"] / M["m00"]

    point = {
        "index": len(calibration_data["points"]) + 1,
        "pixel": {"x": round(pixel_x, 2), "y": round(pixel_y, 2)},
        "robot": {"x": robot_x, "y": robot_y}
    }
    calibration_data["points"].append(point)

    # 新增點位後，重置校正狀態
    calibration_data["calibrated"] = False
    calibration_data["matrix"] = None
    calibration_data["error"] = None

    return {
        "message": f"已新增第 {point['index']} 個校正點 (從輪廓自動抓取)",
        "point": point,
        "point_count": len(calibration_data["points"]),
        "remaining": 9 - len(calibration_data["points"])
    }


@app.delete("/calibration/remove-point")
def api_calibration_remove_point(
    index: int = Query(..., description="要刪除的點位索引 (1-9)")
):
    """
    刪除指定的校正點位
    """
    if index < 1 or index > len(calibration_data["points"]):
        raise HTTPException(status_code=400, detail=f"無效的索引，目前有 {len(calibration_data['points'])} 個點位")

    removed = calibration_data["points"].pop(index - 1)

    # 重新編號
    for i, pt in enumerate(calibration_data["points"]):
        pt["index"] = i + 1

    # 重置校正狀態
    calibration_data["calibrated"] = False
    calibration_data["matrix"] = None
    calibration_data["error"] = None

    return {
        "message": f"已刪除第 {index} 個校正點",
        "removed_point": removed,
        "point_count": len(calibration_data["points"])
    }


@app.post("/calibration/reset")
def api_calibration_reset():
    """
    重置所有校正資料
    """
    calibration_data["points"] = []
    calibration_data["matrix"] = None
    calibration_data["calibrated"] = False
    calibration_data["error"] = None

    return {"message": "校正資料已重置", "point_count": 0}


@app.post("/calibration/calculate")
def api_calibration_calculate():
    """
    根據收集的點位計算仿射變換矩陣
    至少需要 3 個點位，建議使用 9 個點位以獲得更準確的結果
    """
    points = calibration_data["points"]

    if len(points) < 3:
        raise HTTPException(status_code=400, detail=f"至少需要 3 個點位進行校正，目前只有 {len(points)} 個")

    # 準備點位資料
    pixel_points = np.array([[p["pixel"]["x"], p["pixel"]["y"]] for p in points], dtype=np.float32)
    robot_points = np.array([[p["robot"]["x"], p["robot"]["y"]] for p in points], dtype=np.float32)

    try:
        if len(points) == 3:
            # 剛好 3 點，使用 getAffineTransform
            matrix = cv2.getAffineTransform(pixel_points, robot_points)
        else:
            # 多於 3 點，使用最小二乘法估計
            matrix, inliers = cv2.estimateAffine2D(pixel_points, robot_points)
            if matrix is None:
                raise HTTPException(status_code=400, detail="無法計算變換矩陣，請檢查點位資料")

        # 計算校正誤差 (RMSE)
        transformed = []
        for px, py in pixel_points:
            pt = np.array([px, py, 1.0])
            result = matrix @ pt
            transformed.append(result)
        transformed = np.array(transformed)

        errors = np.sqrt(np.sum((transformed - robot_points) ** 2, axis=1))
        rmse = np.sqrt(np.mean(errors ** 2))
        max_error = np.max(errors)

        # 儲存校正結果
        calibration_data["matrix"] = matrix
        calibration_data["calibrated"] = True
        calibration_data["error"] = {
            "rmse": round(float(rmse), 4),
            "max": round(float(max_error), 4),
            "unit": "mm"
        }

        return {
            "message": "校正計算完成",
            "calibrated": True,
            "point_count": len(points),
            "matrix": matrix.tolist(),
            "error": calibration_data["error"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"校正計算失敗: {e}")


@app.post("/calibration/transform")
def api_calibration_transform(
    pixel_x: float = Query(..., description="相機像素座標 X"),
    pixel_y: float = Query(..., description="相機像素座標 Y")
):
    """
    使用校正矩陣將像素座標轉換為手臂座標
    """
    if not calibration_data["calibrated"] or calibration_data["matrix"] is None:
        raise HTTPException(status_code=400, detail="尚未完成校正，請先執行校正計算")

    matrix = calibration_data["matrix"]
    pt = np.array([pixel_x, pixel_y, 1.0])
    result = matrix @ pt

    return {
        "pixel": {"x": pixel_x, "y": pixel_y},
        "robot": {"x": round(float(result[0]), 4), "y": round(float(result[1]), 4)},
        "unit": "mm"
    }


@app.post("/calibration/transform-contour")
def api_calibration_transform_contour(
    contours_id: str = Query(..., description="輪廓 ID"),
    contour_index: int = Query(default=0, description="輪廓索引")
):
    """
    將輪廓中心點轉換為手臂座標
    """
    if not calibration_data["calibrated"] or calibration_data["matrix"] is None:
        raise HTTPException(status_code=400, detail="尚未完成校正，請先執行校正計算")

    if contours_id not in contours_cache:
        raise HTTPException(status_code=404, detail="輪廓資料不存在或已過期")

    cache_data = contours_cache[contours_id]
    contours = json_to_contours(cache_data["data"])

    if contour_index >= len(contours):
        raise HTTPException(status_code=400, detail=f"輪廓索引超出範圍 (共 {len(contours)} 個)")

    cnt = contours[contour_index]
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        raise HTTPException(status_code=400, detail="無法計算輪廓中心點")

    pixel_x = M["m10"] / M["m00"]
    pixel_y = M["m01"] / M["m00"]

    matrix = calibration_data["matrix"]
    pt = np.array([pixel_x, pixel_y, 1.0])
    result = matrix @ pt

    return {
        "contour_index": contour_index,
        "pixel": {"x": round(pixel_x, 2), "y": round(pixel_y, 2)},
        "robot": {"x": round(float(result[0]), 4), "y": round(float(result[1]), 4)},
        "unit": "mm"
    }


@app.post("/calibration/transform-all-contours")
def api_calibration_transform_all_contours(
    contours_id: str = Query(..., description="輪廓 ID")
):
    """
    將所有輪廓中心點轉換為手臂座標
    """
    if not calibration_data["calibrated"] or calibration_data["matrix"] is None:
        raise HTTPException(status_code=400, detail="尚未完成校正，請先執行校正計算")

    if contours_id not in contours_cache:
        raise HTTPException(status_code=404, detail="輪廓資料不存在或已過期")

    cache_data = contours_cache[contours_id]
    contours = json_to_contours(cache_data["data"])

    matrix = calibration_data["matrix"]
    results = []

    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        pixel_x = M["m10"] / M["m00"]
        pixel_y = M["m01"] / M["m00"]

        pt = np.array([pixel_x, pixel_y, 1.0])
        result = matrix @ pt

        results.append({
            "index": i,
            "pixel": {"x": round(pixel_x, 2), "y": round(pixel_y, 2)},
            "robot": {"x": round(float(result[0]), 4), "y": round(float(result[1]), 4)}
        })

    return {
        "count": len(results),
        "unit": "mm",
        "centers": results
    }


# ══════════════════════════════════════════════════════════════════════════════
# MCP Server 設定
# ══════════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    name="vision",
    instructions="工業視覺系統 - 提供相機操作、影像處理、DXF 匯出功能"
)


# ── MCP Tools ────────────────────────────────────────────────────────────────

@mcp.tool()
def camera_connect(ip: str = None) -> str:
    """
    測試 Basler 相機連線

    Args:
        ip: 相機 IP 位址 (例如 "192.168.1.100")

    Returns:
        連線狀態訊息
    """
    cam = BaslerCamera(ip=ip)
    try:
        cam.open()
        return "相機連線成功"
    except Exception as e:
        return f"相機連線失敗: {e}"
    finally:
        cam.close()


@mcp.tool()
def camera_capture(
    output_path: str,
    ip: str = None,
    timeout_ms: int = 5000
) -> str:
    """
    從 Basler 相機拍攝照片並儲存

    Args:
        output_path: 輸出圖片路徑 (例如 "C:/temp/capture.png")
        ip: 相機 IP 位址
        timeout_ms: 超時時間(毫秒)

    Returns:
        儲存的檔案路徑或錯誤訊息
    """
    cam = BaslerCamera(ip=ip)
    try:
        cam.open()
        img = cam.grab_one_bgr(timeout_ms=timeout_ms)
        save_image_file(img, output_path)
        return f"照片已儲存至: {output_path}"
    except Exception as e:
        return f"拍照失敗: {e}"
    finally:
        cam.close()


@mcp.tool()
def threshold_image(
    input_path: str,
    output_path: str,
    threshold_value: int = 100
) -> str:
    """
    對圖片進行二值化處理

    Args:
        input_path: 輸入圖片路徑
        output_path: 輸出圖片路徑
        threshold_value: 閾值 (0-255)

    Returns:
        處理結果訊息
    """
    try:
        img = read_image_file(input_path)
        result = threhold(img, threshold_value)
        save_image_file(result, output_path)
        return f"二值化完成，已儲存至: {output_path}"
    except Exception as e:
        return f"二值化失敗: {e}"


@mcp.tool()
def analyze_connected_components(
    input_path: str,
    output_path: str,
    min_area: int = 20000,
    max_area: int = 300000
) -> str:
    """
    連通元件分析 - 篩選指定面積範圍的物件

    Args:
        input_path: 輸入圖片路徑 (應為二值化圖片)
        output_path: 輸出遮罩圖片路徑
        min_area: 最小面積 (像素)
        max_area: 最大面積 (像素)

    Returns:
        分析結果，包含找到的輪廓數量
    """
    try:
        img = read_image_file(input_path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        filtered_mask, contours = connectedComponentsOnly(img, min_area, max_area)
        save_image_file(filtered_mask, output_path)

        return f"找到 {len(contours)} 個符合條件的輪廓，遮罩已儲存至: {output_path}"
    except Exception as e:
        return f"連通元件分析失敗: {e}"


@mcp.tool()
def erode_mask(
    input_path: str,
    output_path: str,
    kernel_size: int = 11,
    iterations: int = 1
) -> str:
    """
    形態學侵蝕 - 縮小白色區域

    Args:
        input_path: 輸入圖片路徑
        output_path: 輸出圖片路徑
        kernel_size: 侵蝕核心大小 (奇數)
        iterations: 侵蝕次數

    Returns:
        處理結果訊息
    """
    try:
        img = read_image_file(input_path)
        eroded = erode_image(img, kernel_size, iterations)
        save_image_file(eroded, output_path)
        return f"侵蝕完成，已儲存至: {output_path}"
    except Exception as e:
        return f"侵蝕失敗: {e}"


@mcp.tool()
def find_enclosing_circles(input_path: str, min_area: int = 20000, max_area: int = 300000) -> str:
    """
    找出圖片中物件的最小外接圓

    Args:
        input_path: 輸入圖片路徑 (應為二值化圖片)
        min_area: 最小面積篩選
        max_area: 最大面積篩選

    Returns:
        各輪廓的圓心座標和半徑
    """
    try:
        img = read_image_file(input_path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        _, contours = connectedComponentsOnly(img, min_area, max_area)

        results = []
        for i, cnt in enumerate(contours):
            if len(cnt) < 3:
                continue
            center, radius = find_circle(cnt)
            results.append(f"輪廓 {i+1}: 圓心=({center[0]}, {center[1]}), 半徑={radius}")

        if not results:
            return "未找到符合條件的輪廓"

        return "\n".join(results)
    except Exception as e:
        return f"尋找外接圓失敗: {e}"


@mcp.tool()
def export_to_dxf(
    input_path: str,
    output_path: str,
    min_area: int = 20000,
    max_area: int = 300000,
    scale: float = 0.15,
    block_name: str = "PART_CONTOURS"
) -> str:
    """
    從圖片提取輪廓並匯出為 DXF 檔案 (RoboDK 相容)

    Args:
        input_path: 輸入圖片路徑 (應為二值化圖片)
        output_path: 輸出 DXF 檔案路徑
        min_area: 最小面積篩選
        max_area: 最大面積篩選
        scale: 縮放比例 (像素轉毫米)
        block_name: DXF BLOCK 名稱

    Returns:
        匯出結果訊息
    """
    try:
        img = read_image_file(input_path)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        _, contours = connectedComponentsOnly(img, min_area, max_area)

        if not contours:
            return "未找到符合條件的輪廓，無法匯出 DXF"

        contours_to_dxf_as_block(
            contours=contours,
            img_shape=img.shape,
            dxf_path=output_path,
            scale=scale,
            block_name=block_name,
        )

        return f"DXF 已匯出至: {output_path}，包含 {len(contours)} 個輪廓"
    except Exception as e:
        return f"DXF 匯出失敗: {e}"


@mcp.tool()
def full_pipeline(
    input_path: str,
    dxf_output_path: str,
    threshold_value: int = 100,
    min_area: int = 20000,
    max_area: int = 300000,
    scale: float = 0.15
) -> str:
    """
    完整處理流程：讀取圖片 → 二值化 → 連通元件分析 → 匯出 DXF

    Args:
        input_path: 輸入圖片路徑
        dxf_output_path: 輸出 DXF 檔案路徑
        threshold_value: 二值化閾值
        min_area: 最小面積篩選
        max_area: 最大面積篩選
        scale: DXF 縮放比例

    Returns:
        處理結果摘要
    """
    try:
        # 1. 讀取圖片
        img = read_image_file(input_path)

        # 2. 二值化
        binary = threhold(img, threshold_value)

        # 3. 連通元件分析
        _, contours = connectedComponentsOnly(binary, min_area, max_area)

        if not contours:
            return "處理完成，但未找到符合條件的輪廓"

        # 4. 匯出 DXF
        contours_to_dxf_as_block(
            contours=contours,
            img_shape=binary.shape,
            dxf_path=dxf_output_path,
            scale=scale,
        )

        return f"完整流程執行成功！找到 {len(contours)} 個輪廓，DXF 已匯出至: {dxf_output_path}"
    except Exception as e:
        return f"處理流程失敗: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# 啟動函式
# ══════════════════════════════════════════════════════════════════════════════

def run_fastapi(host: str = "0.0.0.0", port: int = 8000):
    """啟動 FastAPI 伺服器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


def run_mcp(host: str = "0.0.0.0", port: int = 8001):
    """啟動 MCP 伺服器 (Streamable HTTP)"""
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


def run_both(host: str = "0.0.0.0", fastapi_port: int = 8000, mcp_port: int = 8001):
    """同時啟動 FastAPI 和 MCP (兩個 HTTP 伺服器)"""
    # FastAPI 在背景執行緒運行
    fastapi_thread = threading.Thread(
        target=run_fastapi,
        args=(host, fastapi_port),
        daemon=True
    )
    fastapi_thread.start()

    print(f"FastAPI 伺服器已啟動於 http://{host}:{fastapi_port}", file=sys.stderr)
    print(f"MCP 伺服器啟動中於 http://{host}:{mcp_port}/mcp", file=sys.stderr)

    # MCP 在主執行緒運行
    run_mcp(host, mcp_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision API + MCP Server")
    parser.add_argument(
        "--mode",
        choices=["both", "fastapi", "mcp"],
        default="both",
        help="運行模式: both (預設), fastapi, mcp"
    )
    parser.add_argument("--host", default="0.0.0.0", help="主機位址")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI 埠號")
    parser.add_argument("--mcp-port", type=int, default=8001, help="MCP 埠號")

    args = parser.parse_args()

    if args.mode == "fastapi":
        print(f"啟動 FastAPI 伺服器於 http://{args.host}:{args.port}")
        run_fastapi(args.host, args.port)
    elif args.mode == "mcp":
        print(f"啟動 MCP 伺服器於 http://{args.host}:{args.mcp_port}/mcp")
        run_mcp(args.host, args.mcp_port)
    else:
        run_both(args.host, args.port, args.mcp_port)
