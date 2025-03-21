import os
import torch
from PIL import ImageDraw
from net3 import *
from data import *
from torchvision.utils import save_image
import time

def cv2Img(img):
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # 检查是否可用GPU
    print(device)
    net = KeypointDetector_v2(num_keypoints=1).to(device)
    weights = 'weights/unet_v2_300.pth'

    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully')
    else:
        print('no loading')
    net.eval()

    for j in os.listdir('test_image'):
        img = Image.open(os.path.join('test_image', j)).convert('RGB')
        img2 = img
        #img2 = img.resize((128, 128))
        img_data = transform(img2)
        img_data = torch.unsqueeze(img_data, dim=0)

        start_time = time.time()  # 记录推理开始时间
        out = net(img_data)
        end_time = time.time()  # 记录推理结束时间

        inference_time = int((end_time - start_time) * 1000)  # 计算推理时间，单位毫秒
        #print("infer_time：", inference_time, "ms")

        out = out.squeeze()
        out = out.unsqueeze(0)

        img = cv2Img(img)
        print(out.shape)
        h, w = np.where(out[0] == out[0].max())
        x = int(w[0] / 1)
        y = int(h[0] / 1)
        cv2.circle(img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)
        print(x,y)
        image_path = f'test_result/{j}'  # 保存路径
        cv2.imwrite(image_path, img)