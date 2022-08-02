## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install

python train.py  --epochs 100 --batch-size 16 --data data.yaml --weights '' --cfg yolov5s.yaml --workers 0

python detect.py --weights C:/Users/shafi/YOLO/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.6 --source C:/Users/shafi/YOLO/test/images
  
python obfuscate.py --weights C:/Users/shafi/YOLO/yolov5/runs/train/exp2/weights/best.pt --img 640 --conf 0.5 --source C:/Users/shafi/YOLO/test/video/2.avi
```

</details>
