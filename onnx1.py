import torch.onnx
from net import UNet
from net3 import KeypointDetector_v2,KeypointDetector_v2_heatmap
import torch

if __name__ == '__main__':

    # 加载权重
    model_path = './weights/unet_v2_heatmap_300.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KeypointDetector_v2_heatmap(num_keypoints=1).to(device)
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    #这是模型的输入维度
    input_data = torch.randn(1, 3, 128, 128, device=device)  # 将通道数从1修改为3

    # 转化为onnx模型
    input_names = ['input']
    output_names = ['output']

    torch.onnx.export(model, input_data, 'unet_v2_heatmap_300.onnx', opset_version=9, verbose=True, input_names=input_names, output_names=output_names)