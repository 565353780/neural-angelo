cd ..
git clone git@github.com:565353780/colmap-manage.git
git clone https://github.com/NVlabs/neuralangelo.git na

cd colmap-manage
./dev_setup.sh

cd ../na

pip install gpustat gdown numpy scipy ipython jupyterlab cython ninja diskcache
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

wandb login cfd3eaa96e9304edc6fc81d46cecc7e749e9bea8
