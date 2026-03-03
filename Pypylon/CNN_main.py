from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import time
import json
import base64
import numpy as np
import ezdxf

def save_labelme_json(result, image_path, save_path, label_map=None):
    """
    將推論結果轉成 LabelMe 格式 JSON
    """
    h, w = result.orig_img.shape[:2]

    # base64 encode image
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    shapes = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = label_map.get(cls_id, f"class_{cls_id}") if label_map else f"class_{cls_id}"

        x1, y1, x2, y2 = box.xyxy[0]
        shape = {
            "label": label,
            "points": [
                [float(x1), float(y1)],
                [float(x2), float(y2)]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }
        shapes.append(shape)

    data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": img_base64
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def train_data():
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11m.pt")
    yaml_path = Path(r"/home/ai/PycharmProjects/Pypylon/CNN_outputs/custom.yaml")

    # 讓 train.txt 裡的相對路徑（1/xxx.png）以 CNN_outputs 為基準
    os.chdir(yaml_path.parent)

    results = model.train(data=yaml_path, epochs=500, imgsz=832,batch=32,device=0)
    #print(results)
def inference_data():
    imgsz = 832

    model_path = r"/home/ai/PycharmProjects/Pypylon/CNN_outputs/runs/detect/train6/weights/best.pt"
    model = YOLO(model_path)
    # 過濾支援的圖片副檔名
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    txt_path = r"/home/ai/PycharmProjects/Pypylon/CNN_outputs/test.txt"
    img_windowns = "Detection Result"
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]  # 去掉空行
        for line in lines:
            line = "/home/ai/PycharmProjects/Pypylon/CNN_outputs/" +line
            results = model(source=line, imgsz=imgsz, conf=0.5)
            result = results[0]
            image = result.plot()
            cv2.namedWindow(img_windowns, cv2.WINDOW_NORMAL)
            # annotated_img = result.plot()
            cv2.imshow(img_windowns, image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
def threhold(img,value):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 確保 dtype 是 uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    _, th = cv2.threshold(img, value, 255, cv2.THRESH_BINARY)
    return th
def connectedComponents(img,min_area,max_area):
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

        # 2) 找外輪廓（要輪廓“裡面內容”，用外輪廓填滿即可）
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

def aa():
    img_path = r"/home/ai/PycharmProjects/Pypylon/CNN_outputs/1/wheel_rim_rotated_102.png"
    img = cv2.imread(img_path)
    data_name = 'wheel_rim_rotated_102'
    # B) 連續抓圖 + 每張交給影像處理副程式
    # cam.grab_loop(on_image=process_image, window_name="preview")
    th_img = threhold(img, 150)
    min_area = 20000  # 依你的圖調整（過濾雜點）
    max_area = 300000
    max_area = 99999999999
    vis_img, contours = connectedComponents(th_img, min_area, max_area)
    dxf_file = contours_to_dxf_as_block(
        contours=contours,
        img_shape=img.shape,
        dxf_path=data_name+".dxf",
        scale=0.15,
        eps_ratio=0.002,
        block_name="PART_CONTOURS",
        layer_name="CONTOUR"
    )
    print("DXF saved:", dxf_file)

if __name__ == "__main__":
    inference_data()
    #train_data()
    #aa()