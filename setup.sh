cd ..
git clone https://github.com/565353780/colmap-manage.git
# git clone --recursive https://github.com/NVlabs/neuralangelo.git na
git clone https://github.com/NVlabs/neuralangelo.git na

cd colmap-manage
./setup.sh

cd ../na

pip install gpustat gdown numpy scipy ipython jupyterlab cython ninja diskcache
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

wandb login
