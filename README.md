# asl_finger model
The base code is available at https://github.com/qqwweee/keras-yolo3 , forked from this base and modified further in this repository.

#### Environment Preparation

The YOLO model used for this runs on Tensorflow 1.6 and Keras 2.1.5 only at this time. So it is recommended to create a seperate python environment (preferably version 3.6) as below. Also the code leverages OpenCV libraries for handling images and videos.

    conda create --name tf1.6 python=3.6 
    conda activate tf1.6


    pip install tensorflow==1.6.0
    pip install keras==2.1.5
    pip install matplotlib
    pip install pillow
    pip install h5py
    pip install opencv-python
    pip install opencv-python-headless


#### Yolo v3 Overview

The YOLO alogorithm was created by researchers at University of Washington (https://pjreddie.com/darknet/yolo/)

- Download the YOLOv3 weights from this website
- Convert the DarkNet YOLO weights to Keras model using convert.py
- You can simply run the detection for the pre trained images detection using yolo_video.py

      wget https://pjreddie.com/media/files/yolov3.weights
      python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
      python yolo_video.py [OPTIONS...] --image, for image detection mode, OR
      python yolo_video.py [video_path] [output_path (optional)]



#### Use --help to see usage of yolo_video.py:

	usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
											 [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
											 [--input] [--output]

	positional arguments:
		--input        Video input path
		--output       Video output path

	optional arguments:
		-h, --help         show this help message and exit
		--model MODEL      path to model weight file, default model_data/yolo.h5
		--anchors ANCHORS  path to anchor definitions, default
											 model_data/yolo_anchors.txt
		--classes CLASSES  path to class definitions, default
											 model_data/coco_classes.txt
		--gpu_num GPU_NUM  Number of GPU to use, default 1
		--image            Image detection mode, will ignore all positional arguments


#### Data Preparation

First step is to generate annotation files and class names file. 
The annotation file must contain one row per image as below.

	image_file_path box1 box2 ... boxN;
	
Where for each box the format should be (No space at the end.)

	x_min,y_min,x_max,y_max,class_id 	

The example file will look like 

	path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
	path/to/img2.jpg 120,300,250,600,2
	...

We used Microsoft VOTT for the annotation , the details can be found @ https://github.com/microsoft/VoTT


#### Model Training

Now modify the train.py to update in the _main() function , also check for other parameters such as learning rate 
and number of epoch , train vs validation split percentage etc

	annotation_path = 'train.txt'
	log_dir = 'logs/000/'
	classes_path = 'model_data/voc_classes.txt'
	anchors_path = 'model_data/yolo_anchors.txt'

also update convereted weights path in the below section for model creation

	model = create_model(input_shape, anchors, num_classes,
							freeze_body=2, weights_path='model_data/yolo_weights.h5')

#### Model Evaluation

#### Model Inference

  ##### Live Video
  
  ##### Recorded Video
  
  ##### Image Prediction
  
