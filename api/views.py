from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models

# Load the trained model
model_path = "photo_processor/models/resnet50_fundus_classifier.pth"  # Update path if needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ResNet50 model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 4)  # Assuming 4 classes

# Load model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class ImageClassificationView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('image')
        if not file:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            image = Image.open(file)
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()

            return Response({"predicted_class": predicted_class}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
