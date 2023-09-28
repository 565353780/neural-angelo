import os
import shutil
from neural_angelo.Method.cmd import runCMD

def trainNA(dataset_folder_path, scene_type='object'):
    assert scene_type in ['indoor', 'outdoor', 'object']

    dataset_name = dataset_folder_path.split('/na/')[0].split('/')[-1]

    cmd = 'cd ../na && python projects/neuralangelo/scripts/generate_config.py' + \
        ' --sequence_name ' + dataset_name + \
        ' --data_dir ' + dataset_folder_path + \
        ' --scene_type ' + scene_type

    if not runCMD(cmd, True):
        print('[ERROR][train::trainNA]')
        print('\t runCMD failed!')
        print('\t cmd:', cmd)
        return False

    yaml_file_path = '../na/projects/neuralangelo/configs/custom/' + \
        dataset_name + '.yaml'
    if not os.path.exists(yaml_file_path):
        print('[ERROR][train::trainNA]')
        print('\t yaml file not exist!')
        print('\t yaml_file_path:', yaml_file_path)
        return False

    output_folder_path = '../neural-angelo/output/' + dataset_name + '/'
    if os.path.exists(output_folder_path):
        shutil.rmtree(output_folder_path)
    os.makedirs(output_folder_path, exist_ok=True)

    new_yaml_file_path = output_folder_path + dataset_name + '.yaml'
    shutil.move(yaml_file_path, new_yaml_file_path)

    cmd = 'cd ../na && torchrun --nproc_per_node=1 train.py' + \
        ' --logdir=' + '../neural-angelo/logs/' + dataset_name + '/' + \
        ' --config=' + new_yaml_file_path + \
        ' --wandb' + \
        ' --wandb_name=' + dataset_name + \
        ' --show_pbar'

    if not runCMD(cmd, True):
        print('[ERROR][train::trainNA]')
        print('\t runCMD failed!')
        print('\t cmd:', cmd)
        return False

    return True
