# 🔥Pytorch-Yanai⚡
## 🔥Base Repository
  * [everybody dance now pytorch](https://github.com/Lotayou/everybody_dance_now_pytorch)
  * [pytorch Realtime Multi-Person Pose Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)

## 🔥Environment
  * Windows
  * Python 3.6
  * CUDA 10.1
  * PyTorch 1.3

## 🔥Everybody Dance Now

### how to make datasets
  * `mkdir datasets/train_A` and `train_B`
  * `ffmpeg -i input.mkv -s 512*288 input_resize.mkv`
  * `ffmpeg -i input_resize.mkv -vcodec png  datasets/train_B/%05d.png`
  * download https://yadi.sk/d/blgmGpDi3PjXvK in `pose-estimator`
  * python ./pose_estimator/compute_coordinates.py -> 60h

### how to test video
  * `mkdir datasets/test_A` and `test_B`
  * `python ./pose_estimator/compute_coordinates_test.py`
  * [Pose2Vid generator(latest)](https://yadi.sk/d/U_sRn9dZiV-G0w) in `./checkpoints/everybody_dance_now_temporal/`
  * `mkdir datasets/own_dance_test/test_sync`
  * `python own_test_video.py --name everybody_dance_now_temporal --model pose2vid --dataroot ./datasets/ --which_epoch latest --netG local --ngf 32 --label_nc 0 --no_instance --resize_or_crop none` : change latest -> 10 or 20...
  * `ffmpeg -r 30 -i datasets/own_dance_test/%05d.png  output.mp4`
  * `rm -rf datasets/own_dance_test/test_sync`

### how to train video
  * `mkdir datasets/test_A` and `test_B` and create pose datasets
  * check datasets
```python
import os
from PIL import Image
error = []
for f in os.listdir():
    try:
        _ = Image.open(f)
    except:
        error.append(f)
error
```
  * `export CUDA_VISIBLE_DEVICES=0,1,2,3` chage your gpu
  * ` > train_512_log.txt & `: delete
  * `tmux` -> `sh scripts/own_test_flow_512` -> Ctrl+B->D


## 🔥OpenPose

### how to install require lib
  * `pip install yacs`
  * [swig](http://www.swig.org/download.html)からinstall -> Path通す
  * `cd lib/pafprocess`
  * `make.sh` or `swig -python -c++ pafprocess.i`->`python setup.py build_ext --inplace`

### how to test demo_pose_web
  * `python demo_pose_web.py --is-gpu`
  * Python3.5の場合ERROR
    * error:`cannot run ‘rc.exe'`->`C:\Program Files (x86)\Windows Kits\10\bin\10.0.15063.0\x64\rc.exe, rcdll.dll を C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin` にコピー
    * error:`Unable to find vcvarsall.bat`-> コンパイラを入れる[参考](https://blog.sky-net.pw/article/25)

## Blender
### how to chroma key
  * install [Blender](https://www.blender.org/download/)
  * open `chromaKey.blend` and move `Laoout tag` and change circle shape
  * move `Compositer tag`, set your video in MovieClip and rendering video
  * run
## TouchDesigner (Coming soon)
### how to start TouchDesigner
  * .toeファイルのディレクトリまでcd
  * `"C:\~~\TouchDesigner099\bin\python.exe" -m venv .venv` -> `.venv\Scripts\activate.bat`
  * [install PyTorch](https://pytorch.org/)
<<<<<<< HEAD
=======

### how to linux
* `du -h -d 3 | sort -hr | head -10` : ファイルの容量のランキング
* `ls -tl` : ファイルの更新時間を表示
* `ls -1 | wc -l` : ファイル数を表示
* `ssh -vvv {hostname}` : sshのエラーログ出力(Permision Denied時)
>>>>>>> 6e677432752f96f29de73d6afd82c0f1b1e1f5a9

### how to git
  * `!git status`
  * `!git init`
  * `!git add .`
  * `!git commit -m "init commit"`

### how to github
  * `git remote add origin https://github.com/YouseiTakei/pytorch-yanai`
  * `git push origin master`  => input github username and password
