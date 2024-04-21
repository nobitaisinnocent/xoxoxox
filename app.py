import torch
import base64
import config
import matplotlib
import numpy as np
from PIL import Image
from io import BytesIO
from train import MnistModel
from flask import Flask, request, render_template, jsonify
matplotlib.use('Agg')

MODEL = None
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def register_hook():
    save_output = SaveOutput()
    hook_handles = []

    for layer in MODEL.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
    return save_output


def module_output_to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()


def mnist_prediction(img):
    save_output = register_hook()
    img = img.to(DEVICE, dtype=torch.float)
    outputs = MODEL(x=img)

    _, output = torch.max(outputs.data, 1)
    pred = module_output_to_numpy(output)
    return pred[0]


@app.route("/process", methods=["GET", "POST"])
def process():
    data_url = str(request.get_data())
    offset = data_url.index(',') + 1
    img_bytes = base64.b64decode(data_url[offset:])
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('L')
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape((1, 28, 28))
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0)

    data = mnist_prediction(img)

    response = {
        'data': str(data)
    }
    return jsonify(response)


@app.route("/", methods=["GET", "POST"])
def start():
    return render_template("default.html")


if __name__ == "__main__":
    MODEL = MnistModel(classes=10)
    MODEL.load_state_dict(torch.load(
        'checkpoint/mnist.pt', map_location=DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)
