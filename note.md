## 安装cv2(opencv-python)
安装过程pip install opencv-python出现问题
anaconda3/lib/python3.5/site-packages/pip/_vendor/pkg_resources/_init_.py
直接去anaconda3/lib/python3.5/site-packages文件夹下
删除多个与pip相关的文件夹,并用重新用conda install安装pip
之后pip install opencv-python安装正常

## 安装pydensecrf
安装过程pip install pydensecrf需要gcc和g++编译
但是由于之前安装cuda和cnn对gcc降级,导致gcc和g++版本不一致
重新对g++降级,查看版本号`gcc --version, g++ --version`
例子:下载并安装gcc/g++ 4.7.x
```
sudo apt-get install -y gcc-4.7
sudo apt-get install -y g++-4.7
```
链接gcc/g++实现降级
```
cd /usr/bin
sudo rm gcc
sudo ln -s gcc-4.7 gcc
sudo rm g++
sudo ln -s g++-4.7 g++
# 查看是否连接到4.7.x
ls –al gcc g++
gcc --version
g++ --version
```
