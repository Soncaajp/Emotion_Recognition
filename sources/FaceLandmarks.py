import cv2
import numpy as np
import tvm
from tvm.contrib import graph_runtime

import os
os.environ["TVM_NUM_THREADS"] = "1"

class LandmarkDetector:
    def __init__(self, model_prefix, normalize = True):
        loaded_json = open(model_prefix+".json").read()
        loaded_lib = tvm.runtime.load_module(model_prefix+".so")
        loaded_params = bytearray(open(model_prefix+".params", "rb").read())
        
        ctx = tvm.cpu()
        self.module = graph_runtime.create(loaded_json, loaded_lib, ctx)
        self.module.load_params(loaded_params)
        
        self.normalize = normalize
        self.input_size = (112,112)
        self.image_mean = np.array([127, 127, 127])
        self.image_std = 128
        
    def _preprocess_img(self, img):                        
        img_preproc = cv2.resize(img, self.input_size)
        if self.normalize:
            img_preproc = (img_preproc - self.image_mean) / self.image_std
        img_preproc = np.transpose(img_preproc, [2, 0, 1])
        img_preproc = np.expand_dims(img_preproc, axis=0)
        img_preproc = img_preproc.astype(np.float32)
        
        return img_preproc
    
    def _process_face_bbox(self, img, bbox):
        '''img must be in BGR format'''
        width = bbox[2]-bbox[0]
        height = bbox[3]-bbox[1]
        
        crop = img[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        crop_preproc = self._preprocess_img(crop)
        
        input_data = tvm.nd.array(crop_preproc)
        self.module.set_input("input", input_data)    
        self.module.run()

        landmarks, quality = self.module.get_output(0).asnumpy()[0], self.module.get_output(1).asnumpy()[0]

        landmarks = [(bbox[0] + width*landmarks[2*i],
                      bbox[1] + height*landmarks[2*i+1]) for i in range(0,len(landmarks)//2)]
        
        quality = np.argmax(quality)
        
        return np.array(landmarks), quality

    def process_bboxes(self, img, bboxes):
        landmarks_list = []
        quality_list = []
        for bbox in bboxes:
            landmarks, quality = self._process_face_bbox(img, bbox)
            landmarks_list.append(landmarks)
            quality_list.append(quality)

        return landmarks_list, quality_list
