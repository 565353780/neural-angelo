import os

from neural_angelo.Method.extract_mesh import extractMesh

data_folder_path = os.environ['HOME'] + '/chLi/Dataset/GS/haizei_1/'
resolution = 2048
block_res = 128

extractMesh(data_folder_path, resolution, block_res)
