from flask import Flask, request, jsonify
from pymongo import MongoClient
from base64 import b64decode

from model.model import shufflenetv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]

        # nn.Sequential将神经网络的层放在一起
        # nn.Conv2d函数
        # in_channels表示的是输入卷积层的图片厚度
        # out_channels表示的是要输出的厚度，即filter的数目
        # kernel_size表示的是卷积核的大小，可以用一个数字表示长宽相等的卷积核，比如kernel_size=3，也可以用不同的数字表示长宽不同的卷积核，比如kernel_size=(3, 2)
        # stride表示卷积核滑动的步长
        # padding表示的是在图片周围填充0的多少，padding=0表示不填充，padding=1四周都填充1维
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128] # 64个卷积核
            nn.BatchNorm2d(64),  # 归一化处理
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64] 第一次Maxpooling之后，图片的长和宽/2

            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]  # 128个卷积核
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32] # 256个卷积核
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]  # 512个卷积核
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]    # 512个卷积核
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]  # 最后输出的cube，之后展开为全连接网络
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)  # 11个类别，11种输出
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


app = Flask('__name__')

client = MongoClient("mongodb://127.0.0.1:27017")

# Create our databases to collect info by WeMos
esdb = client['db']
data = esdb['data']
image = esdb['image']

# Create our databases to collect prediction by Model
predict = esdb['data']


@app.route('/')
def hello_world():
    return '<h1>Group 10 S.A.Y<h1>' + '<h2>we are working hard!!!</h2>'


@app.route("/add", methods=["POST"])
def add():
    # We can get the data POSTED to us using request.get_json()
    try:
        info_data = request.get_json()
        data.insert_one({"temp": info_data["temp"], "ir": info_data["ir"], "lw": info_data["wl"]})

        return "OK", 200

    except Exception as e:
        return "Error inserting record: " + str(e), 500


@app.route("/acquire", methods=["POST"])
def acquire():
    try:
        info_data = request.get_json()
        payload = []
        print(info_data)
        documents = data.find({'ir': info_data['ir']}, {'_id': 0})

        for document in list(documents):
            payload.append(document)

        return jsonify(payload)

    except Exception as e:
        return "Error acquiring record: " + str(e), 500


@app.route("/classify", methods=["POST"])
def image_classify():
    try:
        info_data = request.get_json()

        img_data = b64decode(info_data["image"])
        img = Image.frombytes("RGB", (480, 480), img_data)
        img.show()
        img.save('./image/1.jpg')

        num_classes = 5
        img_size = 480
        width_mult = 1.0
        model = shufflenetv2(n_class=num_classes, input_size=img_size, width_mult=1.0)
        model.load_state_dict(torch.load('./model/model.pth')['state'])
        model.eval()
        img = cv2.imread('./image/1.jpg')
        or_img = img.copy()
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        transform = transforms.Compose([
            transforms.Resize(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        img_tensor = transform(img).unsqueeze(dim=0)
        output = model(img_tensor)
        output = torch.softmax(output, dim=-1)
        score, cls = torch.max(output, dim=-1)

        req = {"prediction": cls.item()+1}

        return jsonify(req)

    except Exception as e:
        return "Error classify image record: " + str(e), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1231)
