from neural_angelo.Method.train import trainNA

def demo():
    data_folder_path = '../colmap-manage/output/3vjia_simple/na/'
    scene_type = 'indoor'

    trainNA(data_folder_path, scene_type)
    return True
