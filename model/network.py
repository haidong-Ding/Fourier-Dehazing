import torch
import torch.nn as nn


class MultiHeadFreqAttention(nn.Module):
    def __init__(self, channel):
        super(MultiHeadFreqAttention, self).__init__()
        self.attn_head_1 = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.attn_head_2 = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.attn_head_3 = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

        self.multi_head_attn = nn.Sequential(
                nn.Conv2d(3, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        a1 = self.attn_head_1(x)
        a2 = self.attn_head_2(x)
        a3 = self.attn_head_3(x)

        a = torch.cat((a1, a2, a3), dim=1)
        a = self.multi_head_attn(a)

        return x * a


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * y


class SALayer(nn.Module):
    def __init__(self, channel):
        super(SALayer, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y


class FourierBlock(nn.Module):
    def __init__(self, channels):
        super(FourierBlock, self).__init__()
        # Frequency domian convolution
        self.layer = nn.Sequential(nn.Conv2d(in_channels=channels*2,
                                             out_channels=channels*2,
                                             kernel_size=1,
                                             stride=1),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(in_channels=channels*2,
                                             out_channels=channels*2,
                                             kernel_size=1,
                                             stride=1),
                                   MultiHeadFreqAttention(channels*2)
                                  )

    def forward(self, x):
        B, C, H, W = x.size()

        # FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        x_ft = torch.stack((x_ft.real, x_ft.imag), dim=-1) # [B, C, H, floor(W/2)+1, 2]

        x_ft = x_ft.permute(0, 1, 4, 2, 3).contiguous() # [B, C, 2, H, floor(W/2)+1]
        x_ft = x_ft.view(B, -1, x_ft.size()[-2], x_ft.size()[-1])

        # Fourier convolution
        out_ft = self.layer(x_ft) # [B, C*2, H, floor(W/2)+1]
        out_ft = out_ft.view(B, C, 2, x_ft.size()[-2], x_ft.size()[-1]).permute(0, 1, 3, 4, 2).contiguous() # [B, C, H, floor(W/2)+1, 2]

        # return to physical space
        out_ft = torch.complex(out_ft[..., 0], out_ft[..., 1])
        output = torch.fft.irfft2(out_ft, dim=(-2, -1), norm='ortho')

        return output


class BasicLayer(nn.Module):
    def __init__(self, channels):
        super(BasicLayer, self).__init__()
        # frequency domian convolution
        self.conv_fourier = FourierBlock(channels)

        # spatial domian convolution
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        # attention mechanism
        self.calayer = CALayer(channels)
        self.salayer = SALayer(channels)

    def forward(self, x):
        # spectral operation
        out_spectral = self.conv_fourier(x)

        # spatial operation
        out_spatial = self.act1(self.conv1(x))
        out_spatial = self.conv2(out_spatial)
        out_spatial = self.calayer(out_spatial)
        out_spatial = self.salayer(out_spatial)

        output = out_spectral + out_spatial + x

        return output


class Net(nn.Module):
    def __init__(self, channels, num_layers):
        super(Net, self).__init__()
        self.pre_conv = nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=True)

        layers = [BasicLayer(channels) for _ in range(num_layers)]
        self.layers = nn.Sequential(*layers)

        self.post_conv = nn.Sequential(nn.Conv2d(channels, channels*2, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(channels*2, 3, kernel_size=3, stride=1, padding=1)
                                       )

    def forward(self, x0):
        x = self.pre_conv(x0)
        x = self.layers(x)
        x = self.post_conv(x)

        return x0 + x

