import ezdxf
import math
from pathlib import Path
from typing import Optional, Tuple, Union, Callable
import cv2
import numpy as np
from pypylon import pylon
import time

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

        # 沒指定就取第一台
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

        # 建議性設定（依相機/介面支援而定，不支援就略過）
        self._safe_set(lambda: self.camera.TriggerSelector.SetValue("FrameStart"))
        self._safe_set(lambda: self.camera.TriggerMode.SetValue("Off"))

        # GigE 常見穩定設定：能設則設，不能設就跳過
        self._safe_set(lambda: self.camera.GevSCPSPacketSize.SetValue(1500))  # PacketSize
        self._safe_set(lambda: self.camera.GevSCPD.SetValue(0))               # Inter-packet delay

        # Host buffer 增加（可視需求調整）
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
        """
        抓一張 BGR 影像（numpy array）。失敗會重試 retries 次。
        """
        if not self.camera or not self.camera.IsOpen():
            self.open()

        assert self.camera is not None

        last_err = None
        for _ in range(retries):
            try:
                # 抓一張：用 StartGrabbingMax(1) + RetrieveResult
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
                # 稍微等一下再重試（可選）
                time.sleep(0.05)

        raise RuntimeError(f"Grab failed: {last_err}")

    def grab_loop(
        self,
        on_image: Callable,                 # on_image(img) -> None
        stop_key: int = 27,                 # ESC
        window_name: Optional[str] = None,  # 例如 "preview"，不想顯示就傳 None
        timeout_ms: int = 5000
    ):
        """
        連續抓圖，把每張 img 丟給 on_image(img) 做後續影像處理。
        可選擇顯示視窗；按 stop_key 中止。
        """
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

                            # 影像處理副程式（你要接的地方）
                            on_image(img)

                            # 顯示（可選）
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
PathLike = Union[str, Path]

def _imread_unicode(path: PathLike, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    """Windows + OpenCV 讀中文路徑：np.fromfile + cv2.imdecode。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"檔案不存在：{p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"OpenCV 無法解碼圖片：{p}")
    return img

def Read_img(input_image_path):
    img = _imread_unicode(input_image_path, cv2.IMREAD_GRAYSCALE)
    return img

def threhold(img,value):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 確保 dtype 是 uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    _, th = cv2.threshold(img, value, 255, cv2.THRESH_BINARY)
    return th

def connectedComponentsOnly(img, min_area, max_area):
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

def connectedComponents(img, min_area, max_area):
    """
    原有函式：連通元件分析 + 侵蝕 + 視覺化（保持向下相容）。
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        img, connectivity=8
    )

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print("label\tarea\tx\ty\tw\th\tcx\tcy")
    kept = []
    all_contours = []

    # 侵蝕參數
    k = 11  # kernel 大小(奇數常見 3/5/7...)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    iters = 1

    # 最終輸出：只放「輪廓內部內容」且侵蝕後的結果（mask）
    eroded_all = np.zeros_like(img, dtype=np.uint8)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if not (min_area <= area <= max_area):
            continue

        # 1) 取出該連通元件的 mask
        comp_mask = ((labels == label).astype(np.uint8)) * 255

        # 2) 找外輪廓（要輪廓"裡面內容"，用外輪廓填滿即可）
        cnts_info = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
        if not contours:
            continue

        # 3) 把輪廓「內部填滿」= 提取輪廓內內容（filled mask）
        filled = np.zeros_like(comp_mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=-1)

        # 4) 對「輪廓內內容」做侵蝕
        eroded = cv2.erode(filled, kernel, iterations=iters)

        # 5) 合併到總結果（只保留侵蝕後的內容）
        eroded_all = cv2.bitwise_or(eroded_all, eroded)

    inside_gray_after_erode = cv2.bitwise_and(img, img, mask=eroded_all)
    # === 新增：侵蝕後「再找輪廓」並畫綠色線 ===
    cnts_info2 = cv2.findContours(eroded_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = cnts_info2[0] if len(cnts_info2) == 2 else cnts_info2[1]


    vis = cv2.cvtColor(inside_gray_after_erode, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours2, -1, (0, 255, 0), 3)  # 綠色線條粗細=3，可自行調

    return vis,contours2
def contours_to_dxf_as_block(contours,img_shape,dxf_path="output_block.dxf",scale=1.0,eps_ratio=0.002,block_name="PART_CONTOURS",layer_name="CONTOUR",):
    """
    將所有輪廓定義為單一 BLOCK,RoboDK 會視為單一物件
    """
    H = img_shape[0]

    # 計算質心
    all_pts = []
    for cnt in contours:
        if cnt is not None and len(cnt) >= 2:
            pts = cnt.reshape(-1, 2)
            all_pts.extend(pts)

    if len(all_pts) == 0:
        raise ValueError("沒有有效的輪廓點")

    all_pts = np.array(all_pts)
    cx_img = np.mean(all_pts[:, 0])
    cy_img = np.mean(all_pts[:, 1])
    cx_dxf = cx_img * scale
    cy_dxf = (H - 1 - cy_img) * scale

    print(f"質心座標 (DXF): ({cx_dxf:.2f}, {cy_dxf:.2f})")

    # 建立 DXF
    doc = ezdxf.new("R2007")
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()

    # 建立圖層
    if layer_name not in doc.layers:
        doc.layers.new(name=layer_name)

    # 建立 BLOCK 定義
    block = doc.blocks.new(name=block_name)

    # 將所有輪廓加入 BLOCK
    for cnt in contours:
        if cnt is None or len(cnt) < 2:
            continue

        peri = cv2.arcLength(cnt, True)
        eps = max(1e-6, eps_ratio * peri)
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2)

        if len(pts) < 2:
            continue

        # 轉換座標 (相對於質心)
        dxf_pts = [
            (
                float(x) * scale - cx_dxf,
                float(H - 1 - y) * scale - cy_dxf
            )
            for x, y in pts
        ]

        # 加入到 BLOCK
        block.add_lwpolyline(
            dxf_pts,
            format="xy",
            close=True,
            dxfattribs={"layer": layer_name},
        )

    # 在模型空間插入 BLOCK (在原點)
    msp.add_blockref(
        block_name,
        insert=(0, 0, 0),  # 插入點
        dxfattribs={
            'xscale': 1.0,
            'yscale': 1.0,
            'rotation': 0
        }
    )

    doc.update_extents()
    doc.audit()
    doc.saveas(dxf_path)

    print(f"已建立 BLOCK: {block_name}")
    return dxf_path

def find_circle(c):
    # 绘制最小外接圆
    (x, y), radius = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(radius)
    return center,radius

# ===== 主程式 =====
if __name__ == "__main__":
    #攝影機
    #影像辨識
    #輸出dxf
    cv2.namedWindow('4.', cv2.WINDOW_NORMAL)
    cam = BaslerCamera(ip="192.168.4.3")
    while True:
        try:
            # A) 只抓一張
            img = cam.grab_one_bgr(timeout_ms=10000, retries=5)
            # B) 連續抓圖 + 每張交給影像處理副程式
            # cam.grab_loop(on_image=process_image, window_name="preview")
            th_img = threhold(img,100)
            min_area = 20000  # 依你的圖調整（過濾雜點）
            max_area = 300000
            #max_area = 99999999999
            vis_img, contours = connectedComponents(th_img, min_area, max_area)
            dxf_file = contours_to_dxf_as_block(
                contours=contours,
                img_shape=img.shape,
                dxf_path="R2004_out_contours1.dxf",
                scale=0.15,
                eps_ratio=0.002,
                block_name="PART_CONTOURS",
                layer_name="CONTOUR"
            )
            print("DXF saved:", dxf_file)

            cv2.imshow('4.', vis_img)
            key = cv2.waitKey(100) & 0xFF
            if key == 27:  # ESC
                break
        except:
            continue
        #finally:
        #    cam.close()
    cv2.destroyAllWindows()

