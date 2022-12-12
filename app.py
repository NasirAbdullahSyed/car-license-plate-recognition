import io, os, torch, cv2, base64
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from ocr_rcnn.CRNN import *
from object_detection.RecognizeCharacter import *
from object_detection.RecognizePlate import *
from ocr_rcnn.Common import *
from flask import Flask, request, redirect, render_template, url_for

# Init our Flask App
app = Flask(__name__)
# Init our class labels
classes = """;Lv4YT"2iNP)MrJj QUh8+RgmaoDI?$ncxtA-W#V/@K!6,:OXFubl0yqwzk_93pf'd*sEBGH17e5S.C(%"""

def load_model():
    params = {
        'imgH': 32,
        'n_classes': len(classes),
        'lr': 0.001,
        'save_dir': 'models',
        'resume': False,
        'cuda': False,
        'schedule': False 
    }
    resume_file = os.path.join(params['save_dir'], 'model_62.ckpt')
    model = CRNN(params)
    model.eval()
    checkpoint = torch.load(resume_file)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

# Loading our model
model = load_model()