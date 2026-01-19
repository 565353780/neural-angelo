import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from neural_angelo.Util.cudnn import init_cudnn
from neural_angelo.Util.set_random_seed import set_random_seed

from neural_angelo.Config.config import Config
from neural_angelo.Module.trainer import Trainer


def demo():
    """主训练函数"""
    print("=" * 60)
    print("Neuralangelo Demo Training")
    print("=" * 60)

    # 小妖怪头
    shape_id = "003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f"
    # 女人上半身
    shape_id = "017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873"
    # 长发男人头
    shape_id = "0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb"

    home = os.environ['HOME']

    data_folder = home + "/chLi/Dataset/pixel_align/" + shape_id + "/"

    checkpoint = data_folder + "na/latest_checkpoint.pt"
    resume = True
    checkpoint = None
    resume = False

    seed = 0

    # 获取配置
    cfg = Config()

    # 设置日志目录
    cfg.logdir = data_folder + "na/logs/"
    cfg.data.root = data_folder + "na/"
    cfg.data.num_images = len(os.listdir(cfg.data.root + 'images/'))

    # 设置随机种子
    set_random_seed(seed, by_rank=False)
    print(f"随机种子: {seed}")

    # 初始化 cuDNN
    init_cudnn(deterministic=False, benchmark=True)

    # 打印关键配置
    print("\n关键配置:")
    print(f"  - 数据集路径: {cfg.data.root}")
    print(f"  - 最大迭代次数: {cfg.max_iter}")
    print(f"  - 训练批量大小: {cfg.data.train.batch_size}")
    print(f"  - 学习率: {cfg.optim.params.lr}")
    print(f"  - 保存检查点间隔: {cfg.checkpoint.save_iter}")
    print(f"  - TensorBoard 记录间隔: {cfg.tensorboard_scalar_iter}")
    print()
    
    # 初始化训练器
    print("初始化训练器...")
    trainer = Trainer(cfg, is_inference=False, seed=seed)
    
    # 设置数据加载器
    print("设置数据加载器...")
    trainer.set_data_loader(cfg, split="train", seed=seed)
    trainer.set_data_loader(cfg, split="val", seed=seed)

    # 加载检查点
    print("加载检查点...")
    trainer.checkpointer.load(
        checkpoint_path=checkpoint,
        resume=resume,
        load_sch=True,
        load_opt=True
    )

    # 初始化 TensorBoard
    trainer.init_tensorboard(cfg, enabled=True)

    # 设置训练模式
    trainer.mode = 'train'

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    trainer.train(
        cfg,
        trainer.train_data_loader,
    )
    
    # 结束训练
    trainer.finalize(cfg)
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"模型和日志已保存到: {cfg.logdir}")
    print("=" * 60)
