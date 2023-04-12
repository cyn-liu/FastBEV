# Step-by-step installation instructions

**a. Clone MSBEVFusion.**

```
git clone https://github.com/xxxxxxxxxx/MSBEVFusion.git
cd MSBEVFusion
```

**b. Create a conda virtual environment and activate it.**

```shell
conda create -n MSBEVFusion python=3.8 -y
conda activate MSBEVFusion
```

**c. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch  -y # 本地
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html # 服务器(cuda 11.7)

# 检查 torch 是否安装成功
python -c 'import torch; print(torch.cuda.is_available())'
```

**d. Install gcc>=5 in conda env (optional).**

```shell
conda install -c omgarcia gcc-6 -y # gcc-6.2
```

**e. Install mmengine and mmcv.**

```shell
pip install -U openmim
mim install mmengine==0.7.0

# 验证安装，如无报错即安装成功。
python -c 'import mmengine;print(mmengine.__version__)'

# install mmcv
mim install mmcv==2.0.0

# 验证安装，如无报错即安装成功。
python -c 'import mmcv; import mmcv.ops'
```

**f. Install mmdet.**

```shell
mim install mmdet==3.0.0rc6

# 验证安装，如无报错即安装成功。
python -c "import mmdet;print(mmdet.__version__)"

```

**g. Install mmdet3d .**

```shell

min install mmdet3d==1.1.0rc3

# 验证安装，如无报错即安装成功。
python -c "import mmdet3d;print(mmdet3d.__version__)"
```

**h. Install mmdetsegmentation .**

```shell

mim install mmsegmentation==1.0.0rc6

# 验证安装，如无报错即安装成功。
python -c "import mmdet3d;print(mmdet3d.__version__)"
```


**i. Install requirements.**

```shell
python -r requirements.txt
```

if find some error of {numba、tensorboard}, please see the instructions in requirements.txt

**j. Prepare pretrained models.**

```shell
mkedir ckpt & cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
cd ..
```



# Quickly verify Python version is appropriate

if run below commands without throwing error,it will be ok

```
pip install numpy==1.22.4
pip install numba==0.53.0
python -c "from numba import jit"
pip install open3d
```

