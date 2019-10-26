from cv2 import cv2
import numpy as np
import torchvision
import torch
from PIL import Image

def transform_image(image):
    """
    Transforms image by using determined transformations.
    Arguments:
        image_bytes(bytearray): Image byte array.
    Returns:
        (Tensor): Transformed image and converted to PyTorch Tensor.
    """
    img_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    return img_transforms(image).unsqueeze(0)

def get_prediction(model, image, names):
    """
    Returns labels and confidence score of passed image.
    Parameters:
        model(torchivsion model): Trained model to pass image to.
        image_bytes(bytearray): Image byte array.
        names(list): Classes of given model.
    Returns:
        pred(list): List of predictions with labels and confidence scores.
    """
    pred = []
    tensor = transform_image(image)
    outputs = model.forward(tensor)
    confidence = torch.nn.functional.softmax(outputs, dim=1).cpu().tolist()
    confidence, names = zip(*sorted(zip(confidence[0], names), reverse=True))

    for idx, conf in enumerate(confidence):
        pred.append({
            'label': names[idx],
            'confidence': conf,
        })

    return pred

def get_model(path, classes):
    """
    Return trained model based on path and classes.
    Parameters:
        path(str): Path to trained model.
        classes(list): List of class names.
    Returns:
        model(torchvision model): Trained model.
    """
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def main():
    # size of camera output
    cam_size = 600

    # label's config
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    text_background = (0, 0, 255)
    text_offset_x = 10
    text_offset_y = cam_size - 25

    # getting all class names
    with open('class_names.txt', 'r') as f:
        class_names = f.read().splitlines()

    # loading the model
    model = get_model('./models/doodle_model.pt', class_names)
    model.eval()

    # starting cv2 video capture
    cap = cv2.VideoCapture(0)
    while True:
        # getting middle of cropped camera output
        crop_size = int(cam_size / 2)

        _, frame = cap.read()
        
        # setting white backgound for lines to draw on to
        img = 255 * np.ones(shape=frame.shape, dtype=np.uint8)
        
        # line detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, maxLineGap=10)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 5)

        # cropping the image for setted output
        mid_h = int(img.shape[0] / 2)
        mid_w = int(img.shape[1] / 2)
        img = img[mid_h-crop_size:mid_h+crop_size, mid_w-crop_size:mid_w+crop_size]

        # converting and normalizing image to array
        # also expanding dims for further keras prediction
        im = Image.fromarray(img, 'RGB')

        # classifying the doodle
        pred = get_prediction(model, im, class_names)

        # generating output text
        text = '{} {}%'.format(pred[0]['label'], int(pred[0]['confidence'] * 100))

        # generating text box    
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))

        # drawing text box, text and showing the lines for better camera adjustment
        cv2.rectangle(img, box_coords[0], box_coords[1], text_background, cv2.FILLED)
        cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 255, 255), thickness=1)
        cv2.imshow('CharAIdes', img)

        key = cv2.waitKey(1)
        if key == 27:
            break

    # ending cv2 cam capture        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    