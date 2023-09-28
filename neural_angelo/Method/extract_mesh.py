import os
import sys
from functools import cmp_to_key
from neural_angelo.Method.cmd import runCMD

sys.path.append('../na/')

def sort_by_id(x, y):
    if x[0] > y[0]:
        return 1
    elif x[0] < y[0]:
        return -1
    return 0


def extractMesh(dataset_folder_path, resolution=2048, block_res=128):
    dataset_name = dataset_folder_path.split('/na/')[0].split('/')[-1]
    output_folder_path = '../neural-angelo/output/' + dataset_name + '/'
    model_file_folder_path = '../neural-angelo/logs/'+ dataset_name + '/'
    yaml_file_path = output_folder_path + dataset_name + '.yaml'
    save_mesh_file_path = output_folder_path + 'mesh.ply'

    model_file_name_list = os.listdir(model_file_folder_path)
    valid_model_idx_list = []
    for model_file_name in model_file_name_list:
        if model_file_name[-3:] != '.pt':
            continue

        model_idx = int(model_file_name.split('_')[1])
        valid_model_idx_list.append([model_idx, model_file_name])

    if len(valid_model_idx_list) == 0:
        print('[ERROR][extract_mesh::extractMesh]')
        print('\t model file not found! please train first and wait!')
        return False

    valid_model_idx_list.sort(key=cmp_to_key(sort_by_id))

    model_file_path = model_file_folder_path + valid_model_idx_list[-1][1]

    print('[INFO][extract_mesh::extractMesh]')
    print('\t start load model...')
    print('\t model_file_path:', model_file_path)

    if not os.path.exists(model_file_path):
        print('[ERROR][extract_mesh::extractMesh]')
        print('\t model file not exist!')
        print('\t model_file_path:', model_file_path)
        return False

    if not os.path.exists(yaml_file_path):
        print('[ERROR][extract_mesh::extractMesh]')
        print('\t yaml file not exist!')
        print('\t yaml_file_path:', yaml_file_path)
        return False

    cmd = 'cd ../na && torchrun --nproc_per_node=1 ' + \
        'projects/neuralangelo/scripts/extract_mesh.py' + \
        ' --config=' + yaml_file_path + \
        ' --checkpoint=' + model_file_path + \
        ' --output_file=' + save_mesh_file_path + \
        ' --resolution=' + str(resolution) + \
        ' --block_res=' + str(block_res)

    if not runCMD(cmd, True):
        print('[ERROR][extract_mesh::extractMesh]')
        print('\t runCMD failed!')
        print('\t cmd:', cmd)
        return False

    return True
