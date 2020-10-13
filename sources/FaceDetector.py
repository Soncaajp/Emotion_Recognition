import cv2
import numpy as np
import tvm
from tvm.contrib import graph_runtime
from lib.nms_np import hard_nms

import os
#os.environ["TVM_NUM_THREADS"] = "1"

class FaceDetector:
    def __init__(self, model_prefix, normalize = True):
        loaded_json = open(model_prefix+".json").read()
        loaded_lib = tvm.runtime.load_module(model_prefix+".so")
        loaded_params = bytearray(open(model_prefix+".params", "rb").read())
        
        ctx = tvm.cpu()
        self.module = graph_runtime.create(loaded_json, loaded_lib, ctx)
        self.module.load_params(loaded_params)
                
        self.input_size = (320,240)
        
        self.normalize = normalize
        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128
        
    def _preprocess_img(self, img):                
        img_preproc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      
        img_preproc = cv2.resize(img_preproc, self.input_size)
        if self.normalize:
            img_preproc = (img_preproc - self.image_mean) / self.image_std
        img_preproc = np.transpose(img_preproc, [2, 0, 1])
        img_preproc = np.expand_dims(img_preproc, axis=0)
        img_preproc = img_preproc.astype(np.float32)
        
        return img_preproc
    
    def _get_predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(box_probs,
                                 iou_threshold=iou_threshold,
                                 top_k=top_k,
                                )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, 4], picked_box_probs[:, :4].astype(np.int32)
        
    def detect(self, img, conf_thresh = 0.5):
        width, height = img.shape[1], img.shape[0]
        img_preproc = self._preprocess_img(img)
        
        #input_data = tvm.nd.array(img_preproc)
        self.module.set_input("input", img_preproc)
        self.module.run()
        
        scores, boxes = self.module.get_output(0).asnumpy(), self.module.get_output(1).asnumpy()
        pred_scores, pred_boxes = self._get_predict(width, height, scores, boxes, conf_thresh)
        return pred_scores, pred_boxes
