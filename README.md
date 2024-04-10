# 3D-BrainTumorSegmentat4MIA
2024 Spring Medical Image Analysis of UCAS

## Environment
```shell 
conda create --name brats python=3.11
source activate brats

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install h5py -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## Run in BsccCloud
Device = NVIDIA A100-PCIE-40GB
```shell
# Run raw data preprocessing
module load anaconda/2021.11
module load cuda/11.8
source activate brats
python raw2hdf5.py

# Run 3D-BrainTumorSegmentat4MIA Model
chmod 777 run.sh
dsub -s run.sh # submit the job
djob           # get the job_id
djob -T job_id # cancel the job
```