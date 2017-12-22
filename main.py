import numpy
import torch
from torch.autograd import Variable
from flask import Flask, jsonify, render_template, request
from PIL import Image
# webapp
app = Flask(__name__)
net = torch.load('net.pkl')     # 载入训练好的模型

def predict_with_pretrain_model(sample):
    s = 1-Variable(
        torch.unsqueeze(
            torch.unsqueeze(torch.from_numpy(sample), dim=0),
            dim=0),
        volatile=True
    ).type(torch.FloatTensor) / 255     # 将28x28的矩阵扩展为1x1x28x28的形式，且为黑底白字
    r = net(s).data.numpy()[0]  # 对图像进行预测
    r = r-numpy.min(r)  # 将结果置为正
    r = r/numpy.sum(r)  # 归一化
    return r.tolist()

@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((numpy.array(request.json, dtype=numpy.uint8))).reshape(28, 28)
    output = predict_with_pretrain_model(input)
    return jsonify(results=output)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
