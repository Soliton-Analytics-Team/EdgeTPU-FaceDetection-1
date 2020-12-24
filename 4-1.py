#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import cv2
from edgetpu.detection.engine import DetectionEngine

CONFIDENCE_TH = 0.6                                                     # 顔検出閾値
MODEL_PATH = "./ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite" # モデルパス
IMAGE_PATH = "./P_20201023_142737_BF.jpg"                               # 入力画像パス

# 画を320x320サイズに縮小・拡大（長方形の場合は余白を黒地にして正方形に変換）
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
    # 検出エンジン初期化
    start = time.time()
    detection_engine = DetectionEngine(MODEL_PATH)
    elapsed_time = time.time() - start
    print ("検出エンジン初期化：{:.6f}".format(elapsed_time) + "[sec]")

    # 入力画像読込み
    start = time.time()
    image = cv2.imread(IMAGE_PATH)
    elapsed_time = time.time() - start
    print ("入力画像読込み：{:.6f}".format(elapsed_time) + "[sec]")

    # 入力画像リサイズ（320x320へ変換）
    start = time.time()
    resized_image = resize_320x320(image)
    elapsed_time = time.time() - start
    print ("入力画像リサイズ：{:.6f}".format(elapsed_time) + "[sec]")

    # TPUによる顔検出
    start = time.time()
    detections = detection_engine.detect_with_input_tensor(
                input_tensor=resized_image.reshape(-1),
                threshold=CONFIDENCE_TH,
                top_k=10
            )
    (h, w) = resized_image.shape[:2]
    for detection in detections:
        box = (detection.bounding_box.flatten().tolist()) * np.array([w, h, w, h])
        (face_left, face_top, face_right, face_bottom) = box.astype('int')
        cv2.rectangle(resized_image, (face_left, face_top),(face_right, face_bottom),(255,0,0),2)
    elapsed_time = time.time() - start
    print ("TPUによる顔検出：{:.6f}".format(elapsed_time) + "[sec]")

    # 検出結果描画
    start = time.time()
    cv2.imshow("Face Detection", resized_image)
    elapsed_time = time.time() - start
    print ("検出結果描画：{:.6f}".format(elapsed_time) + "[sec]")

    # 後処理
    cv2.waitKey(0)
    cv2.destroyAllWindows()