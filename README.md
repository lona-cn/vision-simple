# vision-simple
Inference examples for YOLOv10, YOLOv11, PaddleOCR, and EasyOCR using ONNXRuntime-DirectML and TVM

## HOWTO build
### environment
* [xmake](https://xmake.io) >= 2.9.4
* msvc support c++23
* windows 11
### build
```powershell
./scripts/setupdev-vs.bat
xmake build test_yolo
xmake run test_yolo
```
## demo
### yolo(HellDivers2)
<center>vision-simple</center>

![hd2-yolo-img](doc/images/hd2-yolo.jpg)

<center>ultralytics</center>

![hd2-yolo-gif](doc/images/hd2-yolo.gif)