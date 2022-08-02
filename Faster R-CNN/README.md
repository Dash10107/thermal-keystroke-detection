## How to Run
Install anaconda. Clone or download  the repo, create a folder name my_repo and keep all files from Faster R-CNN folder inside my_repo folder.  Implement following commands in anaconda prompt-

C:\> conda create -n tf_app pip python=3.7

C:\> activate tf_app

(tf_app) C:\>python -m pip install --upgrade pip

(tf_app) C:\>pip install tensorflow-object-detection-api

(tf_app) C:\> pip install flask

(tf_app) C:\> pip install tensorflow==1.13.1

(tf_app) C:\> conda install -c anaconda protobuf

(tf_app) C:\> pip install pillow

(tf_app) C:\> pip install lxml

(tf_app) C:\> pip install Cython

(tf_app) C:\> pip install contextlib2

(tf_app) C:\> pip install jupyter

(tf_app) C:\> pip install matplotlib

(tf_app) C:\> pip install pandas

(tf_app) C:\> pip install opencv-python

Now test the web application by running following commands-

(tf_app) C:\> cd C:\my_repo

(tf_app) C:\my_repo> python main_app.py

Web app can be accessed from http://127.0.0.1:5000/

## Detection and Obfuscation

https://user-images.githubusercontent.com/22468194/182392352-7657e95a-07a4-4de7-b4ba-46e0048b9d81.mp4

The Faster R-CNN is a bit slower for real-time implementation although it offers better precision. Therefore, we adapted YOLO which is much faster and will be suitable for real-time implementation.

