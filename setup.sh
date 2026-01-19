cd ..
git clone https://github.com/565353780/colmap-manage.git
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn

cd colmap-manage
./dev_setup.sh

pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install gpustat gdown numpy scipy ipython jupyterlab \
  cython ninja diskcache

pip install addict gdown gpustat icecream imageio-ffmpeg \
  imutils ipdb k3d kornia lpips matplotlib mediapy \
  nvidia-ml-py3 open3d opencv-python-headless OpenEXR \
  pathlib pillow plotly pyequilib pyexr PyMCubes \
  pyquaternion pyyaml requests scikit-image scikit-video \
  scipy seaborn tensorboard termcolor tqdm trimesh wandb \
  warp-lang

cd ../tiny-cuda-nn
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd bindings/torch
python setup.py install
