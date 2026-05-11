import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()



# =========================
# CNN Model Definition
# =========================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
    nn.Flatten(),
    nn.Linear(64 * 7 * 7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# =========================
# FastAPI App
# =========================
app = FastAPI(title="MNIST CNN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Device Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
model = CNNModel().to(device)

model.load_state_dict(
    torch.load("cnn_mnist_weights.pth", map_location=device)
)

model.eval()

# =========================
# Image Transform
# =========================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# =========================
# Prediction Route
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes))

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {
        "predicted_digit": int(predicted.item())
    }

# =========================
# Run Server
# =========================
if __name__ == "__main__":

    nest_asyncio.apply()

    public_url = ngrok.connect(8000)

    print("Public URL:", public_url)

    uvicorn.run(app, host="0.0.0.0", port=8000)
