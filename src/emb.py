import torch
from torchvision import datasets, transforms
from model.pytorch_facenet import InceptionResnetV1
import numpy as np
from PIL import Image
import cv2


class FaceNet:
    def __init__(self, name_pretrain="vggface2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained=name_pretrain, classify=False).to(self.device)
        self.model.eval()
        self.data_transforms = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor()
        ])

    def get_embedding(self, img):
        im_process = img.copy()
        if isinstance(img, np.ndarray):
            im_process = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        im_process = self.data_transforms(im_process).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb_img = self.model(im_process).to("cpu")
            return emb_img
