import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from neural_angelo.Demo.trainer import demo as demo_train


if __name__ == "__main__":
    demo_train()
