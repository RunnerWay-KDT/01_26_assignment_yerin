import torch
import torch.nn.functional as F


def dice_coef(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Dice coefficient 계산입니다.
    - logits: (B, 1, H, W) 형태의 모델 출력(logits)
    - targets: (B, 1, H, W) 형태의 정답 마스크(0/1)
    """
    probs = torch.sigmoid(logits)
    # 배치 차원만 남기고 평탄화하여 픽셀 단위 계산
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # 교집합과 합집합 계산
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    IoU(Jaccard) 계산입니다.
    - logits: (B, 1, H, W)
    - targets: (B, 1, H, W)
    """
    probs = torch.sigmoid(logits)
    # 배치 차원만 남기고 평탄화
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # 교집합/합집합 계산
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Dice loss (1 - Dice)입니다. 값이 낮을수록 좋습니다."""
    return 1.0 - dice_coef(logits, targets)


def bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits입니다. sigmoid를 내부에서 처리합니다."""
    return F.binary_cross_entropy_with_logits(logits, targets)


__all__ = ["dice_coef", "iou_score", "dice_loss", "bce_loss"]
