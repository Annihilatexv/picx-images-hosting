import numpy as np
from openvino.tools import nncf
from openvino.runtime import Core, serialize

# --- 1. 定义校准数据生成器 (DataLoader) ---
# NNCF 需要一个可迭代对象，每次迭代返回模型输入张量
class RandomDataLoader:
    def __init__(self, input_shape: tuple, num_samples: int = 50):
        # input_shape 应该是 (Batch, Channel, Height, Width) -> (1, 3, 640, 640)
        self.input_shape = input_shape
        self.num_samples = num_samples
        self.current_sample = 0
    
    def __iter__(self):
        # 每次迭代返回一个随机的张量作为输入
        for _ in range(self.num_samples):
            # 生成 (1, 3, 640, 640) 的随机浮点张量，模拟归一化的图片数据
            # 这里的 (0.0, 1.0) 范围是模拟归一化后的图像
            random_input = np.random.uniform(0.0, 1.0, size=self.input_shape).astype(np.float32)
            # NNCF DataLoader 必须返回一个包含输入张量的元组或列表
            yield (random_input,)
            
    def __len__(self):
        return self.num_samples

# --- 2. 配置 ---
IR_MODEL_PATH = "yolov8n_openvino_model/yolov8n.xml"
QUANTIZED_MODEL_PATH = "yolov8n_int8_model.xml"
DEVICE = "CPU"
# YOLOv8n 的输入形状 (Batch=1, Channel=3, Height=640, Width=640)
INPUT_SHAPE = (1, 3, 640, 640) 

# --- 3. 量化执行 ---
ie = Core()

# 1. 读取原始 FP32 OpenVINO 模型
print("加载原始 FP32 模型...")
ov_model = ie.read_model(model=IR_MODEL_PATH)

# 2. 实例化数据加载器
# 这里只用 50 个随机样本进行校准，这速度很快
data_loader = RandomDataLoader(INPUT_SHAPE, num_samples=50) 

# 3. 使用 NNCF 进行量化 (PTQ)
print("开始 NNCF 后训练量化 (PTQ)...")
quantized_model = nncf.quantize(
    model=ov_model, 
    subset=data_loader,
    target_device=nncf.TargetDevice.CPU, # 目标硬件设备
    fast_bias_correction=True # 启用快速偏置校正以提高精度 (可选)
)
print("量化完成。")

# 4. 保存 INT8 模型
serialize(quantized_model, QUANTIZED_MODEL_PATH)
print(f"INT8 模型已保存到: {QUANTIZED_MODEL_PATH} 和 .bin 文件")
