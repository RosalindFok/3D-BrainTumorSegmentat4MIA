# 3D-BrainTumorSegmentat4MIA
2024 Spring Medical Image Analysis of UCAS

## Environment
```shell 
module load anaconda/2021.11 
module load cuda/11.8

conda create --name brats python=3.11

source activate brats
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple/

```