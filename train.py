from neural_angelo.Method.train import trainNA

data_folder_path = '../colmap-manage/output/3vjia_simple/na/'
scene_type = 'outdoor'

trainNA(data_folder_path, scene_type)
