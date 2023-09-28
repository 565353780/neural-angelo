import os
import sys
from neural_angelo.Method.cmd import runCMD

sys.path.append('../na/')

def extractMesh(dataset_folder_path, resolution=2048, block_res=128):
    dataset_name = dataset_folder_path.split('/na/')[0].split('/')[-1]
    output_folder_path = '../neural-angelo/output/' + dataset_name + '/'
    model_file_path = output_folder_path + dataset_name + '.pt'
    yaml_file_path = output_folder_path + dataset_name + '.yaml'
    save_mesh_file_path = output_folder_path + 'mesh.ply'

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
