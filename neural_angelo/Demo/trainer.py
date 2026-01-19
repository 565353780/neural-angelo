import os

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

    checkpoint = data_folder + "na/logs/model_last.pt"

    cfg = Config()
    device = 'cuda:1'
    extract_mesh_only = True

    # 设置日志目录
    cfg.logdir = data_folder + "na/logs/"
    cfg.data.root = data_folder + "na/"
    cfg.data.num_images = len(os.listdir(cfg.data.root + 'images/'))

    # 设置随机种子
    set_random_seed(0)

    # 打印关键配置
    print("\n关键配置:")
    print(f"  - 数据集路径: {cfg.data.root}")
    print(f"  - 每轮迭代次数: {cfg.iters_per_epoch}")
    print(f"  - 最大迭代轮数: {cfg.max_epoch}")
    print(f"  - 训练批量大小: {cfg.data.train.batch_size}")
    print(f"  - 学习率: {cfg.optim.params.lr}")
    print()

    # 初始化训练器
    print("初始化训练器...")
    trainer = Trainer(cfg, device)

    # 加载检查点（如果提供了路径且文件有效，自动恢复训练）
    print("加载检查点...")
    trainer.checkpointer.load(checkpoint)

    if not extract_mesh_only:
        # 开始训练
        print("\n" + "=" * 60)
        print("开始训练...")
        print("=" * 60 + "\n")

        trainer.train()
        trainer.finalize()

    # 导出基本网格
    trainer.exportMeshFile(cfg.logdir + "mesh.ply")

    # 导出高分辨率带纹理的网格，只保留最大连通分量
    trainer.exportMeshFile(
        cfg.logdir + "mesh_textured.ply",
        resolution=1024,
        block_res=128,
        textured=True,
        keep_lcc=False,
    )

    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"模型和日志已保存到: {cfg.logdir}")
    print("=" * 60)
