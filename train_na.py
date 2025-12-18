import os

from neural_angelo.Method.train import trainNA

dataset_root_folder_path = os.environ['HOME'] + '/chLi/Dataset/GS/'
sequence_name = 'haizei_1'
data_folder_path = dataset_root_folder_path + sequence_name + '/'
scene_type = 'object'
gpu_id_list = [2, 3, 4]
master_port = 29512

trainNA(
    sequence_name,
    data_folder_path + 'na/',
    scene_type,
    gpu_id_list,
    master_port,
)
