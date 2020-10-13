import os
import cv2
import glob

class GetVideos:
    def __init__(self, folder, types_file):
        self.folder = folder
        self.types = self._openfile(types_file)
        print(self.types)

    def _openfile(self, types_file):
        with open(types_file, "r") as file:
            return file.readlines()[0].split(',')

    def _getpathes(self, type):
        return glob.glob(self.folder + '/*.' + type)
    
    def _createVid(self, path):
        return cv2.VideoCapture(path)

    def _processVideo(self, vid, function, info, writeMethod, filename):
        ret = True
        while ret:
            ret, frame = vid.read()
            if ret:
                info.append(function(frame))
        writeMethod(info, filename)

    def _onerun(self, type, function, writeMethod):
        for filename in self._getpathes(type):
            info = []
            vid = self._createVid(filename)
            self._processVideo(vid, function, info, writeMethod, filename)

    def run(self, function, writeMethod):
        for type in self.types:
            self._onerun(type, function, writeMethod)
