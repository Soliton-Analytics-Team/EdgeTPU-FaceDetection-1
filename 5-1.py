#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sys
from edgetpu.detection.engine import DetectionEngine

CONFIDENCE_TH = 0.6
MODEL_PATH = './ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite'
detection_engine = DetectionEngine(MODEL_PATH)

def resize_320x320(image):
    size = 320
    interpolation = cv2.INTER_AREA
    (h, w) = image.shape[:2]

    #入力画像が正方形ならそのままresize
    if h == w:
        return cv2.resize(image, (size, size), interpolation)

    #入力画像が長方形ならば余白を入れて正方形に変換してからresize
    mask_size = h if h > w else w
    channel = None if len(image.shape) < 3 else image.shape[2]
    if channel is None:
        mask = np.zeros((mask_size, mask_size), dtype=image.dtype)
        mask[:h, :w] = image[:h, :w]
    else:
        mask = np.zeros((mask_size, mask_size, channel), dtype=image.dtype)
        mask[:h, :w, :] = image[:h, :w, :]

    return cv2.resize(mask, (size, size), interpolation)

if __name__ == '__main__':
    camera_id = 1
    delay = 1
    window_name = "Face Detection"

    # カメラ画像読込み準備
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit()

    while True:
        # カメラ画像１枚読込み
        ret, frame = cap.read()

        # W320xH320へサイズ変換
        resized_image = resize_320x320(frame)

        # 顔の検出
        detections = detection_engine.detect_with_input_tensor(
            input_tensor=resized_image.reshape(-1),
            threshold=CONFIDENCE_TH,
            top_k=10
        )

        # 顔が検出されなければメッセージ出力
        if not detections:
            print("no face was detected.")

        # 検出BOXの描画（複数検出にも対応）
        (h, w) = resized_image.shape[:2]
        for detection in detections:
            box = (detection.bounding_box.flatten().tolist()) * np.array([w, h, w, h])
            (face_left, face_top, face_right, face_bottom) = box.astype('int')
            cv2.rectangle(resized_image, (face_left, face_top), (face_right, face_bottom), (255,0,0), 2)

        # リサイズ後カメラ画像を検出BOX込みで表示
        cv2.imshow(window_name, resized_image)

        # 何かキー入力があれば終了
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # 後処理
    cv2.destroyWindow(window_name)