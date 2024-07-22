import torch.nn as nn
import torch.nn.functional as F
import torch

class conv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        + Instantiate modules: conv-relu-norm
        + Assign them as member variables
        """
        super(conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.relu = nn.PReLU()
        # with learnable parameters
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class conv3d_x3(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x3, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.conv_3 = conv3d(out_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_3 = self.conv_3(self.conv_2(z_1))
        return z_3 + self.skip_connection(x)

class conv3d_x2(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x2, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.conv_2 = conv3d(out_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        z_2 = self.conv_2(z_1)
        return z_2 + self.skip_connection(x)


class conv3d_x1(nn.Module):
    """Three serial convs with a residual connection.
    Structure:
        inputs --> ① --> ② --> ③ --> outputs
                   ↓ --> add--> ↑
    """

    def __init__(self, in_channels, out_channels):
        super(conv3d_x1, self).__init__()
        self.conv_1 = conv3d(in_channels, out_channels)
        self.skip_connection=nn.Conv3d(in_channels,out_channels,1)

    def forward(self, x):
        z_1 = self.conv_1(x)
        return z_1 + self.skip_connection(x)

class deconv3d_x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x3, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x3 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,5,1,2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels,5,1,2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels,5,1,2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv),dim=1) 
        return self.conv_x3(rhs_add)+ rhs_up

class deconv3d_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x2, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x2= nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,5,1,2),
            nn.PReLU(),
            nn.Conv3d(out_channels, out_channels,5,1,2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv),dim=1) 
        return self.conv_x2(rhs_add)+ rhs_up

class deconv3d_x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv3d_x1, self).__init__()
        self.up = deconv3d_as_up(in_channels, out_channels, 2, 2)
        self.lhs_conv = conv3d(out_channels // 2, out_channels)
        self.conv_x1 = nn.Sequential(
            nn.Conv3d(2*out_channels, out_channels,5,1,2),
            nn.PReLU(),
        )

    def forward(self, lhs, rhs):
        rhs_up = self.up(rhs)
        lhs_conv = self.lhs_conv(lhs)
        rhs_add = torch.cat((rhs_up, lhs_conv),dim=1) 
        return self.conv_x1(rhs_add)+ rhs_up
        

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
        """Output with shape [batch_size, 1, depth, height, width]."""
        # Do NOT add normalize layer, or its values vanish.
        y_conv = self.conv_2(self.conv_1(x))
        return nn.Sigmoid()(y_conv)


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        self.conv_1 = conv3d_x1(1, 16)
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

        self.out = softmax_out(32, 1)

    def forward(self, x):
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
        return self.out(deconv)
    
    
# class CombinedModel(nn.Module):
#     def __init__(self):
#         super(CombinedModel, self).__init__()
#         # self.vnet = VNet()
#         self.vnet = SimpleVNet()
#         self.diag_fc = nn.Linear(4, 64)  # 四个诊断特征，输出64个特征
        
#         # 假设 VNet 的输出是 128 * 32 * 32 * 32 （在 VNet 修改后计算得到的）
#         vnet_output_features = 128 * 32 * 32 * 32
#         combined_features = vnet_output_features + 64  # 加上诊断特征
        
#         self.fc_combine = nn.Linear(combined_features, 1024)
#         self.final_conv = nn.Conv3d(1024, 1, kernel_size=1, stride=1)
#         self.sigmoid = nn.Sigmoid()  # 输出层，用于得到分割图

#     def forward(self, mri, diag):
#         mri_features = self.vnet(mri)
#         mri_features_flat = mri_features.view(mri_features.size(0), -1)
#         diag_features = F.relu(self.diag_fc(diag))
#         combined = torch.cat((mri_features_flat, diag_features), dim=1)
#         # combined = F.relu(self.fc_combine(combined))
#         combined = combined.view(-1, 1024, 32, 32, 32)  # 根据 VNet 的具体输出调整尺寸
#         output = self.final_conv(combined)
#         output = self.sigmoid(output)  # 使用 sigmoid 使输出归一化到 [0,1]
#         return output
    
class VNetWithDiagnosis(nn.Module):
    def __init__(self):
        super(VNetWithDiagnosis, self).__init__()
        self.initial_downsample = nn.Conv3d(1, 1, kernel_size=3, stride=2, padding=1)
        self.encoder1 = self.make_layers(1, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self.make_layers(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.decoder1 = self.make_layers(32 + 4, 16)
        self.upsample1 = nn.ConvTranspose3d(16, 16, kernel_size=2, stride=2)
        self.decoder2 = self.make_layers(48, 16)
        self.final_conv = nn.Conv3d(32, 1, kernel_size=1)

    def make_layers(self, in_channels, out_channels, final_layer=False):
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if final_layer:
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=1))
        return nn.Sequential(*layers)

    def forward(self, x, diagnosis):
        original_size = x.size()[2:]  # Store the original size of the input
        x = self.initial_downsample(x)
        x1 = self.encoder1(x)
        x = self.pool1(x1)
        x2 = self.encoder2(x)
        x = self.pool2(x2)

        diagnosis = diagnosis.view(1, 4, 1, 1, 1).expand(-1, -1, x.size(2), x.size(3), x.size(4))
        
        
#         if diagnosis.dim() == 1 and diagnosis.size(0) == 4:
#             diagnosis = diagnosis.unsqueeze(0)  # 将形状从 (4,) 改为 (1, 4)

#         # 使用expand确保diagnosis张量可以与x在空间维度上匹配
#         diagnosis = diagnosis.view(1, 4, 1, 1, 1).expand(-1, -1, x.size(2), x.size(3), x.size(4))

        x = torch.cat((x, diagnosis), 1)

        x = self.decoder1(x)
        x = self.upsample1(x)

        # Adjust x to match dimensions of x2 if they are close
        if abs(x.size(2) - x2.size(2)) <= 1:
            x = F.interpolate(x, size=x2.shape[2:], mode='trilinear', align_corners=False)

        x = torch.cat((x, x2), 1)
        x = self.decoder2(x)

        x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)

        # Again, adjust x to match dimensions of x1 if they are close
        if abs(x.size(2) - x1.size(2)) <= 1:
            x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)

        x = torch.cat((x, x1), 1)
        x = self.final_conv(x)

        # Upsample to the original input size
        x = F.interpolate(x, size=original_size, mode='trilinear', align_corners=False)
        return x

