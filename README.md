# Pytorch-Yanai
### how to make datasets
### how to test video
### how to train video
### how to train own video
### how to test own video

### how to test demo_pose_web
  * `pip install yacs`
  * [swig](http://www.swig.org/download.html)からinstall -> Path通す
  * `cd lib/pafprocess`
  * `make.sh` or `swig -python -c++ pafprocess.i`->`python setup.py build_ext --inplace`
  * `python demo_pose_web.py --is-gpu`
  * Python3.5の場合ERROR
    * error:`cannot run ‘rc.exe'`->C:\Program Files (x86)\Windows Kits\10\bin\10.0.15063.0\x64\rc.exe, rcdll.dll を C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin にコピー
    * error:`Unable to find vcvarsall.bat`-> コンパイラを入れる[参考](https://blog.sky-net.pw/article/25)


### how to start TouchDesigner
  * .toeファイルのディレクトリまでcd
  * `"C:\~~\TouchDesigner099\bin\python.exe" -m venv .venv` -> `.venv\Scripts\activate.bat`
  * [install PyTorch](https://pytorch.org/)
  *

### how to git
  * `!git status`(指定したデータが無視されているか確認)
  * `!git init`
  * `!git add .`
  * `!git commit -m "init commit"`

### how to github
  * `git remote add origin https://github.com/YouseiTakei/pytorch-yanai`
  * `git push origin master`  => input github username and password
