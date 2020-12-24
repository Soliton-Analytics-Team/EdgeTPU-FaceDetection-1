#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time
import cv2
import face_recognition

IMAGE_PATH = "./P_20201023_142737_BF.jpg"       # 入力画像パス

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

    # dlibによる顔検出
    start = time.time()
    boxes = face_recognition.face_locations(resized_image, model='hog')		# HOGアルゴリズムの場合はこちらを使う
    #boxes = face_recognition.face_locations(resized_image, model='cnn')	# DeepLearningの場合はこちらを使う
    for box in boxes:
        (face_top, face_right, face_bottom, face_left) = box
        cv2.rectangle(resized_image, (face_left, face_top),(face_right, face_bottom),(255,0,0),2)
    elapsed_time = time.time() - start
    print ("dlibによる顔検出：{:.6f}".format(elapsed_time) + "[sec]")

    # 検出結果描画
    start = time.time()
    cv2.imshow("Face Detection", resized_image)
    elapsed_time = time.time() - start
    print ("検出結果描画：{:.6f}".format(elapsed_time) + "[sec]")

    # 後処理
    cv2.waitKey(0)
    cv2.destroyAllWindows()