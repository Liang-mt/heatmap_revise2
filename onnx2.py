import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import time

def cv2Img(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def infer_onnx_new_modle(onnx_model_path, image_path, output_path):
    # 加载 ONNX 模型
    session = ort.InferenceSession(onnx_model_path)

    # 准备输入数据
    img = Image.open(image_path).convert('RGB').resize((128, 128))
    img_array = np.array(img)
    img_array = img_array.transpose(2, 0, 1)
    img_data = img_array.astype(np.float32) / 255
    input_data = np.expand_dims(img_data, axis=0)

    # 执行推理
    start_time = time.time()
    output = session.run(None, {'input': input_data})
    end_time = time.time()

    inference_time = int((end_time - start_time) * 1000)
    print("infer_time：", inference_time, "ms")

    # 后处理
    out = output[0].squeeze(0)
    print(f"Shape of 'out': {out.shape}")  # 打印 out 的形状，用于调试

    img = cv2Img(img)
    for i in range(1):  # 假设只处理第一个输出
        h, w = np.where(out[i] == out[i].max())
        x = int(w[0])
        y = int(h[0])
        print(f"Keypoint coordinates: ({x}, {y})")
        cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

    # 保存输出图像
    cv2.imwrite(output_path, img)

def infer_onnx_old_modle(onnx_model_path, image_path, output_path):
    # 加载 ONNX 模型
    session = ort.InferenceSession(onnx_model_path)

    # 准备输入数据（替代 torchvision.transforms.ToTensor）
    img = Image.open(image_path).convert('RGB').resize((80, 80))
    img_array = np.array(img)                     # 转为 HWC 格式的 numpy 数组
    img_array = img_array.transpose(2, 0, 1)      # 转为 CHW 格式
    img_data = img_array.astype(np.float32) / 255 # 归一化到 [0,1]

    # 调整输入维度为 [1, 3, 80, 80]
    input_data = np.expand_dims(img_data, axis=0)  # 添加 batch 维度 [1, 3, 80, 80]
    #input_data = np.tile(input_data, (2, 1, 1, 1)) # 复制到 [2, 3, 80, 80]

    # 执行推理
    start_time = time.time()
    output = session.run(None, {'input': input_data})
    end_time = time.time()

    inference_time = int((end_time - start_time) * 1000)
    print("infer_time：", inference_time, "ms")

    # 后处理（直接使用 NumPy 替代 torch 操作）
    out = output[0].squeeze(0)

    # 绘制结果
    img = cv2Img(img)
    for i in range(1):  # 假设只处理第一个输出
        h, w = np.where(out[i] == out[i].max())
        x = int(w[0] * 0.8)
        y = int(h[0] * 0.8)
        print(f"Keypoint coordinates: ({x}, {y})")
        cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)

    # 保存输出图像
    cv2.imwrite(output_path, img)

if __name__ == '__main__':
    onnx_model_path = './unet_v2_heatmap_300.onnx'
    input_image_path = './test_result/577.png'
    output_image_path = 'output_image.jpg'
    infer_onnx_new_modle(onnx_model_path, input_image_path, output_image_path)