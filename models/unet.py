import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) * 2 블록입니다.
    - padding=1을 사용하여 입력/출력 공간 크기를 유지합니다.
    - U-Net의 기본 특징 추출 단위로 반복적으로 사용됩니다.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # 연속된 3x3 Conv 두 번으로 지역적 특징을 점진적으로 강화합니다.
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """
    U-Net (padding=1 버전) 구현입니다.
    - Encoder: 다운샘플링하며 전역 의미 정보를 추출합니다.
    - Decoder: 업샘플링하며 위치 정보를 복원합니다.
    - Skip connection: 동일 해상도 특징을 연결하여 경계 정보를 보존합니다.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()  # Encoder의 DoubleConv 블록들
        self.ups = nn.ModuleList()    # Decoder의 UpConv + DoubleConv 블록들
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 해상도 절반으로 축소

        # Encoder 경로: 채널 수를 점진적으로 증가시키며 특징을 추출합니다.
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck: 가장 압축된 표현을 학습하는 구간입니다.
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder 경로: 업샘플링 후 skip feature와 결합합니다.
        rev = list(reversed(features))
        up_in = features[-1] * 2
        for f in rev:
            # UpConv로 해상도 2배 확장
            self.ups.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            # Skip 연결 후 DoubleConv로 특징 정제
            self.ups.append(DoubleConv(up_in, f))
            up_in = f

        # 최종 1x1 Conv: 채널을 클래스 수로 변환
        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: 각 단계의 특징을 skip으로 저장
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        # Decoder: 업샘플링 후 skip feature와 concat
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]

            # 크기 불일치가 있을 경우 보정 (padding 사용 시에도 안전장치)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            # 채널 방향으로 skip feature 결합
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        # 출력은 logits (sigmoid는 loss/metric에서 적용)
        return self.final(x)


__all__ = ["DoubleConv", "UNet"]
