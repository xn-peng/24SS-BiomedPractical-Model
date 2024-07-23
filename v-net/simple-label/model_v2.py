import torch
import torch.nn as nn

# Origianl VNet definition here comes from https://github.com/Lee-Wayne/VNet-Pytorch

class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.PReLU()
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class conv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.conv_3 = conv3d(out_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_3 + self.skip_connection(x)

class conv3d_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv3d_x2, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_2 = self.conv_2(z_1)
        return z_2 + self.skip_connection(x)

class conv3d_x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv3d_x1, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.skip_connection = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        return z_1 + self.skip_connection(x)

class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x3 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x3(rhs_add) + rhs_up

class deconv3d_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x2, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x2 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x2(rhs_add) + rhs_up

class deconv3d_x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x1, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x1 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels, 5, 1, 2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv), dim=1)
        return self.conv_x1(rhs_add) + rhs_up

def conv3d_as_pool(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=0),
        nn.PReLU())

def deconv3d_as_up(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride),
        nn.PReLU()
    )

class softmax_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(softmax_out, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        y_conv = self.conv_2(self.conv_1(x))
        return nn.Sigmoid()(y_conv)

def average_pooling_downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.AvgPool3d(kernel_size=2, stride=2),
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.PReLU()
    )

class VNet(nn.Module):
    def __init__(self, use_downsampling=False):
        super(VNet, self).__init__()
        self.use_downsampling = use_downsampling
        self.initial_downsample = nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.PReLU()
        )
        self.conv_1 = conv3d_x1(16, 16) if use_downsampling else conv3d_x1(1, 16)
        self.pool_1 = conv3d_as_pool(16, 32)
        self.conv_2 = conv3d_x2(32, 32)
        self.pool_2 = conv3d_as_pool(32, 64)
        self.conv_3 = conv3d_x3(64, 64)
        self.pool_3 = conv3d_as_pool(64, 128)
        self.conv_4 = conv3d_x3(128, 128)
        self.pool_4 = conv3d_as_pool(128, 256)
        self.bottom = conv3d_x3(256, 256)
        self.deconv_4 = deconv3d_x3(256, 256)
        self.deconv_3 = deconv3d_x3(256, 128)
        self.deconv_2 = deconv3d_x2(128, 64)
        self.deconv_1 = deconv3d_x1(64, 32)
        self.final_upsample = nn.ConvTranspose3d(32, 32, kernel_size=4, stride=2, padding=1)
        self.out = softmax_out(32, 1)

    def forward(self, x):
        if self.use_downsampling:
            x = self.initial_downsample(x)
        conv_1 = self.conv_1(x)
        pool = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool)
        pool = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool)
        pool = self.pool_3(conv_3)
        conv_4 = self.conv_4(pool)
        pool = self.pool_4(conv_4)
        bottom = self.bottom(pool)
        deconv = self.deconv_4(conv_4, bottom)
        deconv = self.deconv_3(conv_3, deconv)
        deconv = self.deconv_2(conv_2, deconv)
        deconv = self.deconv_1(conv_1, deconv)
        if self.use_downsampling:
            deconv = self.final_upsample(deconv)
        return self.out(deconv)

class ExtraFeatures(nn.Module):
    def __init__(self, num_extra_features, output_channels):
        super(ExtraFeatures, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_extra_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, spatial_dims):
        features = self.fc(features)
        features = features.view(-1, features.size(1), 1, 1, 1).expand(-1, -1, *spatial_dims)
        return features

class CombinedVNetModel(nn.Module):
    def __init__(self, num_extra_features, use_downsampling=False):
        super(CombinedVNetModel, self).__init__()
        self.initial_downsample = nn.Conv3d(1, 1, kernel_size=3, stride=2, padding=1)
        self.vnet = VNet(use_downsampling)
        self.extra_features = ExtraFeatures(num_extra_features, 32)
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        self.fc = nn.Linear(1, 1)  # Final fully connected layer for classification

    def forward(self, x, extra_features):
        vnet_output = self.vnet(x)
        extra_features_output = self.extra_features(extra_features, vnet_output.shape[2:])
        combined_output = vnet_output + extra_features_output
        output = torch.sigmoid(self.final_conv(combined_output))
        pooled_output = self.global_avg_pool(output)  # Apply Global Average Pooling
        pooled_output = pooled_output.view(pooled_output.size(0), -1)  # Flatten the output
        final_output = self.fc(pooled_output)  # Pass through the final fully connected layer
        return final_output
