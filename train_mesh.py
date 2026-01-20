import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from neural_angelo.Demo.mesh_trainer import demo as demo_train_mesh


if __name__ == "__main__":
    demo_train_mesh()
