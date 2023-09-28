from neural_angelo.Method.train import trainNA

def demo():
    data_folder_path = '../colmap-manage/output/3vjia_simple/na/'

    trainNA(data_folder_path)
    return True
