from neural_angelo.Method.extract_mesh import extractMesh

data_folder_path = '../colmap-manage/output/3vjia_simple/na/'
resolution = 2048
block_res = 128

extractMesh(data_folder_path, resolution, block_res)
