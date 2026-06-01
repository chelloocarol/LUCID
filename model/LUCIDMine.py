import torch
import torch.nn as nn
import torch.nn.functional as F

from .Student import Student


class MineVisibilityPriorExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def to_image_space(self, image):
        # CoA uses CLIP-normalized inputs; priors should be measured in RGB image space.
        return (image * self.clip_std + self.clip_mean).clamp(0, 1)

    def forward(self, image, size):
        image = self.to_image_space(image)
        red = image[:, 0:1]
        green = image[:, 1:2]
        blue = image[:, 2:3]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        dark_channel = image.min(dim=1, keepdim=True)[0]
        glare = (image.max(dim=1, keepdim=True)[0] - luminance).clamp(0, 1)

        local_mean = F.avg_pool2d(F.pad(luminance, (3, 3, 3, 3), mode="reflect"), 7, stride=1)
        low_visibility = (1.0 - (luminance - local_mean).abs() * 4.0).clamp(0, 1)
        priors = torch.cat([luminance, dark_channel, glare, low_visibility], dim=1)
        return F.interpolate(priors, size=size, mode="bilinear", align_corners=False)


class VisibilityConditionedCoAAdapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))
        self.project = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )
        self._init_identity()

    def _init_identity(self):
        convs = [module for module in self.project if isinstance(module, nn.Conv2d)]
        nn.init.kaiming_uniform_(convs[0].weight, a=0.2)
        nn.init.zeros_(convs[0].bias)
        nn.init.normal_(convs[1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(convs[1].bias)

    def forward(self, feature, priors):
        glare = priors[:, 2:3]
        low_visibility = priors[:, 3:4]
        reliability = (0.55 + 0.45 * low_visibility) * (1.0 - 0.35 * glare)
        return feature + self.gate * self.project(priors) * reliability


class GlareAwareResidualCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))
        self.project = nn.Conv2d(4, 3, kernel_size=1)
        nn.init.normal_(self.project.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.project.bias)

    def forward(self, image, restored, priors):
        glare = priors[:, 2:3]
        visibility = priors[:, 3:4]
        residual = restored - image
        learned_scale = torch.tanh(self.project(priors))
        residual_mask = (1.0 - glare).clamp(0, 1) * (0.5 + 0.5 * visibility)
        residual_scale = (1.0 + self.gate * learned_scale * residual_mask).clamp(0.55, 1.45)
        return image + residual * residual_scale


class LUCIDMine(Student):
    def __init__(self, res_blocks=1):
        super().__init__(res_blocks=res_blocks)
        self.mine_prior = MineVisibilityPriorExtractor()
        self.visibility_adapter = VisibilityConditionedCoAAdapter(128)
        self.glare_calibrator = GlareAwareResidualCalibrator()

    def forward(self, x):
        ini = x
        res1x = self.conv_input(x)
        res1x_1, res1x_2 = res1x.split([(res1x.size()[1] // 2), (res1x.size()[1] // 2)], dim=1)
        feature_mem = [res1x_1]
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion1(res2x_1, feature_mem)
        res2x_2 = self.conv1(res2x_2)
        feature_mem.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.dense1(res2x) + res2x

        res4x = self.conv4x(res2x)
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion2(res4x_1, feature_mem)
        res4x_2 = self.conv2(res4x_2)
        feature_mem.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion3(res8x_1, feature_mem)
        res8x_2 = self.conv3(res8x_2)
        feature_mem.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        res16x_1 = self.fusion4(res16x_1, feature_mem)
        res16x_2 = self.conv4(res16x_2)
        res16x = torch.cat((res16x_1, res16x_2), dim=1)
        res16x = self.visibility_adapter(res16x, self.mine_prior(ini, res16x.shape[2:]))

        res2xx = res2x
        res4xx = res4x
        res8xx = res8x
        res16xx = res16x

        res_dehaze = res16x
        in_ft = res16x * 2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
        feature_mem_up = [res16x_1]

        res16x = self.convd16x(res16x)
        res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        res8x = self.convd8x(res8x)
        res8x = F.interpolate(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        res4x = self.convd4x(res4x)
        res4x = F.interpolate(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)

        res2x = self.dense_2(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2(res2x_2)

        feature_mem_up.append(res2x_1)
        res2x = torch.cat((res2x_1, res2x_2), dim=1)
        res2x = self.convd2x(res2x)
        res2x = F.interpolate(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x
        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
        x_1 = self.fusion_1(x_1, feature_mem_up)
        x_2 = self.conv_1(x_2)
        x = torch.cat((x_1, x_2), dim=1)

        x = self.conv_output(x)
        x = self.glare_calibrator(self.mine_prior.to_image_space(ini), x, self.mine_prior(ini, ini.shape[2:]))

        return x, [res2xx, res4xx, res8xx, res16xx]
