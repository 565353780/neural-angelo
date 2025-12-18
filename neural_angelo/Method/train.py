import os
import shutil
from neural_angelo.Method.cmd import runCMD

def trainNA(
    sequence_name: str,
    dataset_folder_path: str,
    scene_type: str='object',
    gpu_id_list: list=[0],
    master_port: int=29500,
) -> bool:
    assert scene_type in ['indoor', 'outdoor', 'object']

    cmd = 'python ./projects/neuralangelo/scripts/generate_config.py' + \
        ' --sequence_name ' + sequence_name + \
        ' --data_dir ' + dataset_folder_path + \
        ' --scene_type ' + scene_type

    if not runCMD(cmd, True):
        print('[ERROR][train::trainNA]')
        print('\t runCMD failed!')
        print('\t cmd:', cmd)
        return False

    yaml_file_path = './projects/neuralangelo/configs/custom/' + \
        sequence_name + '.yaml'
    if not os.path.exists(yaml_file_path):
        print('[ERROR][train::trainNA]')
        print('\t yaml file not exist!')
        print('\t yaml_file_path:', yaml_file_path)
        return False

    output_folder_path = './output/' + sequence_name + '/'
    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    new_yaml_file_path = output_folder_path + sequence_name + '.yaml'
    shutil.move(yaml_file_path, new_yaml_file_path)

    gpu_ids = ",".join(map(str, gpu_id_list))
    nproc_per_node = str(len(gpu_id_list))

    cmd = 'CUDA_VISIBLE_DEVICES=' + gpu_ids + \
        ' torchrun --nproc_per_node=' + nproc_per_node + \
        ' --master_port ' + str(master_port) + \
        ' train.py' + \
        ' --logdir=' + './logs/' + sequence_name + '/' + \
        ' --config=' + new_yaml_file_path + \
        ' --show_pbar'

    if not runCMD(cmd, True):
        print('[ERROR][train::trainNA]')
        print('\t runCMD failed!')
        print('\t cmd:', cmd)
        return False

    return True
