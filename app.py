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

def transform_licensce_img(img):
    pil_img = img
    old_width, old_height = pil_img.size
    aspect_ratio = old_width / old_height
    width = round(32 * aspect_ratio)
    transform =  transforms.Compose([transforms.Resize(size=(32, width)), transforms.Grayscale(1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return transform(pil_img).unsqueeze(0)

def get_ocr_prediction(classes, model, img_tensor):
    converter = LabelCodec(classes)
    logits = model(img_tensor).transpose(1, 0)
    logits = torch.nn.functional.log_softmax(logits, 2)
    logits = logits.contiguous().cpu()
    T, B, H = logits.size()
    pred_sizes = torch.LongTensor([T for i in range(B)])
    probs, pos = logits.max(2)
    pos = pos.transpose(1, 0).contiguous().view(-1)
    result = converter.decode(pos.data, pred_sizes.data, raw=False)
    return result

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

# Our API Routes
@app.route('/')
def home():
    result_image = request.args.get('result_image') if 'result_image' in request.args else "<img src='/static/default.png' alt='Result' class='block w-[300px] h-[275px] p-6' />"
    result_text = request.args.get('result_text') if 'result_text' in request.args else "0000-0000"
    return render_template('home.html', res_img = result_image, res_txt = result_text)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        plate_img, bounded_img = extractPlate(cv_img)
        plate_img = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        bounded_img = Image.fromarray(cv2.cvtColor(bounded_img, cv2.COLOR_BGR2RGB))
        plate_img = transform_licensce_img(plate_img)
        result_text = get_ocr_prediction(classes, model, plate_img)
        buffered = io.BytesIO()
        bounded_img.save(buffered, format="JPEG")
        result_img = base64.b64encode(buffered.getvalue()).decode('ascii')
        img_tag = f'<img src="data:image/jpeg;base64,{result_img}" alt="Result" class="block w-[300px] h-[275px] p-6" />'
        return redirect(url_for('home', result_text=result_text, result_image=img_tag))

# Utility Functions
def drawBoxAroundPlate(image, plate):
    p2fRectPoints = cv2.boxPoints(plate.plateLocation)
    p2fRectPoints = p2fRectPoints.astype(int)
    cv2.line(image, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), (0.0, 255.0, 0.0), 2)
    cv2.line(image, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), (0.0, 255.0, 0.0), 2)
    cv2.line(image, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), (0.0, 255.0, 0.0), 2)
    cv2.line(image, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), (0.0, 255.0, 0.0), 2)

def extractPlate(img):
    possiblePlates = detectPlatesInScene(img)
    possiblePlates = detectCharsInPlates(possiblePlates)
    if len(possiblePlates) == 0:
        return False
    else:                                                   
        possiblePlates.sort(key = lambda possiblePlate: possiblePlate.strChars, reverse = True)
        plate = possiblePlates[0]
        drawBoxAroundPlate(img, plate)
    return plate.imgPlate, img