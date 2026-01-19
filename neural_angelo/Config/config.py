class Config:
    """Neuralangelo 配置类，包含所有超参数
    
    所有配置项都已内置默认值，无需用户输入即可运行训练。
    """
    
    # ==================== 基础训练配置 ====================
    logging_iter: int = 9999999999999  # 禁用打印日志
    max_iter: int = 500000
    max_epoch: int = 9999999999  # 最大 epoch 数
    tensorboard_scalar_iter: int = 100
    tensorboard_image_iter: int = 100
    image_save_iter: int = 200  # 图像保存迭代频率
    validation_iter: int = 1000
    validation_epoch: int = 9999999999999  # 禁用基于 epoch 的验证
    speed_benchmark: bool = False
    timeout_period: int = 9999999  # 超时时间（秒）
    nvtx_profile: bool = False  # NVTX 性能分析
    pretrained_weight: str = None  # 预训练权重路径
    
    # 指标计算频率（如果为 None，则从 checkpoint 配置复制）
    metrics_iter: int = None
    metrics_epoch: int = None
    
    # ==================== CUDNN 配置 ====================
    class Cudnn:
        benchmark: bool = True
        deterministic: bool = False
    
    cudnn = Cudnn()
    
    # ==================== 检查点配置 ====================
    class Checkpoint:
        save_iter: int = 100  # 每 N 次迭代保存检查点
        save_epoch: int = 9999999999  # 每 N 个 epoch 保存检查点
        save_latest_iter: int = 200  # 每 N 次迭代保存 latest_checkpoint.pt
        save_period: int = 9999999999  # 基于时间的保存周期（分钟）
        strict_resume: bool = True  # 严格模式加载 state_dict
    
    checkpoint = Checkpoint()
    
    # ==================== 训练器配置 ====================
    class Trainer:
        type: str = "projects.neuralangelo.trainer"
        depth_vis_scale: float = 0.5
        grad_accum_iter: int = 1  # 梯度累积迭代次数
        image_to_tensorboard: bool = True  # 是否将图像写入 TensorBoard
        
        class EMAConfig:
            enabled: bool = False
            load_ema_checkpoint: bool = False
            beta: float = 0.9999  # EMA 衰减系数
            start_iteration: int = 0  # 开始 EMA 的迭代次数
        
        ema_config = EMAConfig()
        
        class DDPConfig:
            find_unused_parameters: bool = False
            static_graph: bool = True
        
        ddp_config = DDPConfig()
        
        class LossWeight:
            render: float = 1.0
            eikonal: float = 0.1
            curvature: float = 5e-4
        
        loss_weight = LossWeight()
        
        class Init:
            type: str = "none"
            gain: float = None
        
        init = Init()
        
        class AMPConfig:
            enabled: bool = False
            init_scale: float = 65536.0
            growth_factor: float = 2.0
            backoff_factor: float = 0.5
            growth_interval: int = 2000
        
        amp_config = AMPConfig()
    
    trainer = Trainer()
    
    # ==================== 模型配置 ====================
    class Model:
        type: str = "projects.neuralangelo.model"
        
        class Object:
            class SDF:
                class MLP:
                    num_layers: int = 1
                    hidden_dim: int = 256
                    skip: list = []
                    activ: str = "softplus"
                    
                    class ActivParams:
                        beta: int = 100
                    
                    activ_params = ActivParams()
                    geometric_init: bool = True
                    weight_norm: bool = True
                    out_bias: float = 0.5
                    inside_out: bool = False
                
                mlp = MLP()
                
                class Encoding:
                    type: str = "hashgrid"
                    levels: int = 16
                    
                    class Hashgrid:
                        min_logres: int = 5
                        max_logres: int = 11
                        dict_size: int = 22
                        dim: int = 8
                        range: list = [-2, 2]
                    
                    hashgrid = Hashgrid()
                    
                    class Coarse2Fine:
                        enabled: bool = True
                        init_active_level: int = 4
                        step: int = 5000
                    
                    coarse2fine = Coarse2Fine()
                
                encoding = Encoding()
                
                class Gradient:
                    mode: str = "numerical"
                    taps: int = 4
                
                gradient = Gradient()
            
            sdf = SDF()
            
            class RGB:
                class MLP:
                    num_layers: int = 4
                    hidden_dim: int = 256
                    skip: list = []
                    activ: str = "relu_"
                    activ_params: dict = {}
                    weight_norm: bool = True
                
                mlp = MLP()
                mode: str = "idr"
                
                class EncodingView:
                    type: str = "spherical"
                    levels: int = 3
                
                encoding_view = EncodingView()
            
            rgb = RGB()
            
            class SVar:
                init_val: float = 3.0
                anneal_end: float = 0.1
            
            s_var = SVar()
        
        object = Object()
        
        class Background:
            enabled: bool = True
            white: bool = False
            view_dep: bool = True
            
            class MLP:
                num_layers: int = 8
                hidden_dim: int = 256
                skip: list = [4]
                num_layers_rgb: int = 2
                hidden_dim_rgb: int = 128
                skip_rgb: list = []
                activ: str = "relu"
                activ_params: dict = {}
                activ_density: str = "softplus"
                activ_density_params: dict = {}
            
            mlp = MLP()
            
            class Encoding:
                type: str = "fourier"
                levels: int = 10
            
            encoding = Encoding()
            
            class EncodingView:
                type: str = "spherical"
                levels: int = 3
            
            encoding_view = EncodingView()
        
        background = Background()
        
        class Render:
            rand_rays: int = 512
            
            class NumSamples:
                coarse: int = 64
                fine: int = 16
                background: int = 32
            
            num_samples = NumSamples()
            num_sample_hierarchy: int = 4
            stratified: bool = True
        
        render = Render()
        
        class AppearEmbed:
            enabled: bool = False
            dim: int = 8
        
        appear_embed = AppearEmbed()
    
    model = Model()
    
    # ==================== 优化器配置 ====================
    class Optim:
        type: str = "AdamW"
        fused_opt: bool = False  # 是否使用融合优化器
        
        class Params:
            lr: float = 1e-3
            weight_decay: float = 1e-2
        
        params = Params()
        
        class Sched:
            iteration_mode: bool = True
            type: str = "two_steps_with_warmup"
            warm_up_end: int = 5000
            two_steps: list = [300000, 400000]
            step_size: int = 9999999999  # 步进大小
            gamma: float = 10.0
        
        sched = Sched()
    
    optim = Optim()
    
    # ==================== 数据配置 ====================
    class Data:
        name: str = "dummy"  # 数据集名称
        type: str = "projects.neuralangelo.data"
        root: str = "datasets/nerf-synthetic/lego"
        use_multi_epoch_loader: bool = True
        num_workers: int = 4
        preload: bool = True
        num_images: int = None  # 训练图像数量

        class Train:
            image_size: list = [1024, 1024]
            batch_size: int = 2
            subset: int = None

        train = Train()

        class Val:
            image_size: list = [300, 300]
            batch_size: int = 2
            subset: int = 4
            max_viz_samples: int = 4

        val = Val()

        class Readjust:
            center: list = [0.0, 0.0, 0.0]
            scale: float = 1.0

        readjust = Readjust()

    data = Data()
    
    # ==================== 测试数据配置 ====================
    class TestData:
        name: str = "dummy"
        num_workers: int = 0
        type: str = "imaginaire.datasets.images"
        
        class Test:
            batch_size: int = 1
            is_lmdb: bool = False
            roots: str = None
        
        test = Test()
    
    test_data = TestData()

    # 动态属性（在运行时设置）
    logdir = None
    local_rank: int = 0  # 本地 GPU rank
    inference_args: dict = {}  # 推理参数
    
    def setdefault(self, key: str, value):
        """设置默认值，如果属性不存在则创建"""
        if not hasattr(self, key):
            setattr(self, key, value)
        return getattr(self, key)
    
    def to_dict(self):
        """将配置转换为字典格式"""
        def _to_dict(obj):
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, list):
                return [_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            else:
                result = {}
                for key in dir(obj):
                    if not key.startswith('_') and not callable(getattr(obj, key)):
                        value = getattr(obj, key)
                        result[key] = _to_dict(value)
                return result
        
        return _to_dict(self)
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict) and isinstance(getattr(self, key), object):
                    # 递归更新嵌套对象
                    nested_obj = getattr(self, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_obj, nested_key):
                            setattr(nested_obj, nested_key, nested_value)
                else:
                    setattr(self, key, value)
            else:
                setattr(self, key, value)
        return self
    
    def __repr__(self):
        """返回配置的字符串表示"""
        return f"Config(data.root={self.data.root}, max_iter={self.max_iter})"
