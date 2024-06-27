eval "$(/pollard/home/sdrusinsky/miniforge3/bin/conda shell.bash hook)"
source /pollard/home/sdrusinsky/miniforge3/bin/activate enformer_ft
conda update --all

conda \
    install -y \
    -c pytorch \
    -c nvidia \
    pytorch=2.3.1 \
    pytorch-cuda=12.1

conda \
  install -y \
  -c conda-forge \
  cython \
  ipython \
  matplotlib \
  pandas \
  scikit-learn \
  seaborn \
  pyarrow \
  tqdm \
  lightning \
  mkl \ 
  rich \
  torchmetrics \ 


pip install enformer_pytorch kipoiseq pysam wandb vcfpy
