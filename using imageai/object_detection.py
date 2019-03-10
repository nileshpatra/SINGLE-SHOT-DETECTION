import os
from imageai.Detection import ObjectDetection

directory = os.path.dirname(os.path.abspath('__file__'))
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(directory + '/resnet50_coco_best_v2.0.1.h5')
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image = directory+'/image.jpeg' , 
	output_image_path = directory + '/newimage.jpeg')

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )