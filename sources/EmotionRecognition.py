import cv2
import numpy as np
import tvm
from tvm.contrib import graph_runtime

import os
os.environ["TVM_NUM_THREADS"] = "1"

class EmotionRecognition:
    def __init__(self, model_prefix, normalize = True):
        loaded_json = open(model_prefix+".json").read()
        loaded_lib = tvm.runtime.load_module(model_prefix+".so")
        loaded_params = bytearray(open(model_prefix+".params", "rb").read())
        
        ctx = tvm.cpu()
        self.module = graph_runtime.create(loaded_json, loaded_lib, ctx)
        self.module.load_params(loaded_params)
        
        self.normalize = normalize
        self.input_size = (98,98)
        self.labels = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Anxiety', 'Disgust','Anger']
        
    def _preprocess_img(self, img):                        
        img_preproc = cv2.resize(img, self.input_size)
        if self.normalize:
            img_preproc = img_preproc / 255
        img_preproc = np.transpose(img_preproc, [2, 0, 1])
        img_preproc = np.expand_dims(img_preproc, axis=0)
        img_preproc = img_preproc.astype(np.float32)
        
        return img_preproc

    def run(self, img):
        crop_preproc = self._preprocess_img(img)
    
        input_data = tvm.nd.array(crop_preproc)
        self.module.set_input("input_1", input_data)
        self.module.run()

        return {self.labels[i] : str(r) for i, r in enumerate(self.module.get_output(0).asnumpy()[0])}
