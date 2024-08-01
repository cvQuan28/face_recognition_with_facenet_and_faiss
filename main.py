import torch
from torchvision import datasets, transforms
from model.pytorch_facenet import InceptionResnetV1
import cv2
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
model.eval()

data_transforms = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

img = cv2.imread(r"E:\Data\Faces\V1092470\V1092470_120.jpg")
img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
img = data_transforms(img).unsqueeze(0).to(device)

with torch.no_grad():
    emb_img = model(img).to(device)
