from torch import nn

# Define the Custom Neural Network
class CustomNet(nn.Module):
    def __init__(self):

        super(CustomNet, self).__init__()

        # Input Shape: [B, 3, 224, 224]

        # Define the Layers of the Neural Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2) # [B, 64, 112, 112]
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # [B, 128, 56, 56]
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2) # [B, 256, 28, 28]
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2) # [B, 512, 14, 14]
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=2) # [B, 1024, 7, 7]

        # Flatten ALL Dimension, BUT Starting from the 3rd (2+1):
        # FROM [B, 1024, 7, 7] TO [B, 1024, 49]
        self.flatten = nn.Flatten(2)

        self.fc1 = nn.Linear(1024, 200)
    

    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu()

        # Average EACH 1024-Channel Vector
        # over the 49 Spatial Positions
        # FROM [B, 1024, 49] TO [B, 1024]
        x = self.flatten(x).mean(-1)

        return self.fc1(x)