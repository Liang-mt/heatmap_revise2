import os

from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net2 import *
from net3 import *
from network.HRNet import *
from torchvision.utils import save_image

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight="./params/unet_50.pth"
train_data_path=r'./datasets/data_center_train.txt'
val_data_path=r'./datasets/data_center_val.txt'
save_path='train_image'

train_data_size = len(MyDataset(train_data_path))
test_data_size = len(MyDataset(val_data_path))

#设置训练网络的一些参数
epoch = 500



if __name__ == '__main__':

    print("训练数据集的长度为：{} ".format(train_data_size))
    print("测试数据集的长度为：{} ".format(test_data_size))

    train_dataloader = DataLoader(MyDataset(train_data_path), batch_size=16, shuffle=True)
    test_dataloader = DataLoader(MyDataset(val_data_path), batch_size=16, shuffle=True)

    #net = UNet().to(device)
    net = KeypointDetector_v2(1).to(device)
    if os.path.exists(weight):
        net.load_state_dict(torch.load(weight))
        print('successful load weight！')
    else:
        print('not successful load weight')
    optimizer = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    for i in range(epoch):
        print("------------第 {} 轮训练开始--------------".format(i + 1))

        for number, (image, segment_image) in enumerate(train_dataloader):
            imgs, targets = image.to(device), segment_image.to(device)
            outputs = net(imgs)

            loss = loss_fun(outputs, targets)



            # pre,tar = cal_pre2tar(outputs, targets)
            # print(pre,tar)

            #oks = calculate_oks(pre, tar)  # 计算OKS得分

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if number % 5 == 0:
                print(f'{i + 1}-{number}-train_loss===>>{loss.item()}')
            if number % 1 == 0:
                _image = imgs[0]
                _segment_image = targets[0]
                _out_image = outputs[0]
                img = torch.stack([_segment_image[0], _out_image[0]], dim=0)
                img = torch.unsqueeze(img, dim=1)
                save_image(img, f'{save_path}/{number}.png', nrow=5)

        total_test_loss = 0

        with torch.no_grad():
            total_rmse = 0
            for n, (image, segment_image) in enumerate(test_dataloader):
                imgs, targets = image.to(device), segment_image.to(device)
                outputs = net(imgs)
                loss = loss_fun(outputs, targets)
                total_test_loss += loss.item()


        print("整体测试数据集的Loss:{}".format(total_test_loss))


        if (i + 1) % 100 == 0:
            weight_path = f'./weights/unet_v2_{i + 1}.pth'  # 构建正确的文件路径

            checkpoint_last = {}  # 将这里改为字典
            checkpoint_last["model"] = net.state_dict()
            checkpoint_last["optimizer"] = optimizer.state_dict()
            checkpoint_last["epoch"] = i

            torch.save(net.state_dict(), weight_path)  # 保存模型
            print("模型已保存")