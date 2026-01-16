import os

from neural_angelo.Method.train import trainNA

home = os.environ['HOME']

# 小妖怪头
SHAPE_ID="003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f"
# 女人上半身
SHAPE_ID="017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873"
# 长发男人头
SHAPE_ID="0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb"

dataset_root_folder_path = home + "/chLi/Dataset/pixel_align/" + SHAPE_ID + '/'
data_folder_path = dataset_root_folder_path + 'na/'
scene_type = 'object'
gpu_id_list = [0, 1]
master_port = 29512

trainNA(
    SHAPE_ID,
    data_folder_path,
    scene_type,
    gpu_id_list,
    master_port,
)
