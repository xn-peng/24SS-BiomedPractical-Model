import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, num_extra_features=3):
        super(UNet, self).__init__()

        self.enc_conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(2)

        self.enc_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(2)

        self.enc_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.extra_features_fc = nn.Sequential(
            nn.Linear(num_extra_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()  # 添加Sigmoid激活函数

    def forward(self, x, extra_features):
        # 编码路径
        enc1 = self.enc_conv1(x)
        enc2 = self.pool1(enc1)
        enc3 = self.enc_conv2(enc2)
        enc4 = self.pool2(enc3)
        enc5 = self.enc_conv3(enc4)
        enc6 = self.pool3(enc5)
        enc7 = self.enc_conv4(enc6)
        bottleneck = self.pool4(enc7)
        bottleneck = self.bottleneck(bottleneck)

        # Decode
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc7), dim=1)
        dec4 = self.dec_conv4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc5), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec_conv1(dec1)

        #  Add extra features
        extra_out = self.extra_features_fc(extra_features)
        extra_out = extra_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        extra_out = extra_out.expand(dec1.size(0), -1, dec1.size(2), dec1.size(3), dec1.size(4))

        combined = dec1 + extra_out

        output = self.final_conv(combined)
        output = self.sigmoid(output)
        return output
