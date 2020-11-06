# -*- coding: utf-8 -*-
# @Time : 2020/10/30 9:58 下午
# @Author : SongWeinan
# @Software: PyCharm
# 欲买桂花同载酒，终不似、少年游。
# ======================================================================================================================
import numpy as np
import cv2
from shapely.geometry import Polygon
from structures.polygons import Polygons
from process.postprocess import BasePostprocess
import pyclipper
import time
import matplotlib.pylab as plt


class DBPostprocess(BasePostprocess):
    def __init__(self, cfg):
        self.min_size = 5
        self.unclip_ratio = cfg.POSTPROCESS.DB.UNCLIP_RATIO
        self.max_candidates = 1000
        self.box_thresh = cfg.POSTPROCESS.DB.BOX_THRESH
        self.binary_thresh = cfg.POSTPROCESS.DB.BINARY_THRESH

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2
        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def unclip(self, box):
        poly = Polygon(box)
        distance = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours, ), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores

    def _decode(self, binary_map: 'torch.Tensor'):

        result_rboxes = []
        segmentation = binary_map > self.binary_thresh
        # plt.figure("img")
        # plt.figure(figsize=(12, 12))
        # plt.imshow(segmentation)
        # plt.show()
        tmp_boxes, tmp_scores = self.boxes_from_bitmap(
            binary_map, segmentation, segmentation.shape[1], segmentation.shape[0])
        for k in range(len(tmp_boxes)):
            if tmp_scores[k] > self.box_thresh:
                result_rboxes.append(tmp_boxes[k])
        polygons = Polygons(result_rboxes)
        return polygons

    def __call__(self, outputs: dict, params: dict):
        binary_maps = outputs[0].to('cpu').numpy()

        batch_size = len(binary_maps)
        batch_results = []

        for idx in range(batch_size):
            binary_map = binary_maps[idx]
            polygons = self._decode(binary_map[0])
            self._rescale(polygons, params[idx])
            batch_results.append(polygons)
        return batch_results