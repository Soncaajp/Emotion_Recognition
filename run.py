from lib.emo_pipeline import EmoPipeline
from lib.getVideo import GetVideos
import json

video_folder = './videos'
types_file = './video_types.txt'

getVideo = GetVideos(video_folder, types_file)
emoPipeline = EmoPipeline()

def write_method(result, filename):
	result = {str(i):r for i, r in enumerate(result)}
	print(result)
	with open('./results/' + filename.split('/')[-1] + '.json', 'w') as file:
		json.dump(result, file)

def processing(image):
	return emoPipeline.run(image)

getVideo.run(processing, write_method)
