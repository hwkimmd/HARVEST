"""Fixed cross-view DINO student/teacher used by HARVEST."""
from __future__ import annotations

import math

import torch
from torch import nn

from harvest.dinov3_adapter import activate_dinov3

OUT_DIM = 8192
STUDENT_TEMPERATURE = 0.1
TEACHER_TEMPERATURE_START = 0.04
TEACHER_TEMPERATURE_END = 0.07
TEACHER_TEMPERATURE_WARMUP_FRACTION = 0.30
CENTER_MOMENTUM = 0.9
EMA_START = 0.994
EMA_END = 1.0
KOLEO_WEIGHT = 0.1


class BackboneWithDinoHead(nn.Module):
    def __init__(self, backbone, embed_dim, dinov3_source):
        super().__init__()
        activate_dinov3(dinov3_source)
        from dinov3.layers.dino_head import DINOHead

        self.backbone = backbone
        self.head = DINOHead(
            in_dim=embed_dim,
            out_dim=OUT_DIM,
            hidden_dim=2048,
            bottleneck_dim=256,
            nlayers=3,
        )

    def forward(self, images):
        features = self.backbone.forward_features(images)["x_norm_clstoken"]
        return features, self.head(features)


class HarvestDomainSSLTrainer(nn.Module):
    def __init__(self, build_backbone, embed_dim, dinov3_source):
        super().__init__()
        activate_dinov3(dinov3_source)
        from dinov3.loss import DINOLoss, KoLeoLoss

        self.student = BackboneWithDinoHead(
            build_backbone(), embed_dim, dinov3_source
        )
        self.teacher = BackboneWithDinoHead(
            build_backbone(), embed_dim, dinov3_source
        )
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher.requires_grad_(False).eval()
        self.center_loss = DINOLoss(
            out_dim=OUT_DIM,
            student_temp=STUDENT_TEMPERATURE,
            center_momentum=CENTER_MOMENTUM,
        )
        self.center_loss.init_weights()
        self.koleo_loss = KoLeoLoss()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @staticmethod
    def teacher_temperature(progress):
        if progress >= TEACHER_TEMPERATURE_WARMUP_FRACTION:
            return TEACHER_TEMPERATURE_END
        return TEACHER_TEMPERATURE_START + (
            progress / TEACHER_TEMPERATURE_WARMUP_FRACTION
        ) * (TEACHER_TEMPERATURE_END - TEACHER_TEMPERATURE_START)

    @staticmethod
    def ema_momentum(progress):
        cosine = (math.cos(math.pi * progress) + 1.0) / 2.0
        return EMA_END - (EMA_END - EMA_START) * cosine

    @torch.no_grad()
    def update_teacher(self, momentum):
        for student, teacher in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            teacher.mul_(momentum).add_(student.detach(), alpha=1.0 - momentum)
        for student, teacher in zip(
            self.student.buffers(), self.teacher.buffers()
        ):
            teacher.copy_(student)

    def forward(self, global_crops, local_crops, progress):
        if global_crops.shape[0] % 2:
            raise ValueError("global crop batch must contain two equal views")
        batch = global_crops.shape[0] // 2
        with torch.no_grad():
            _, teacher_logits = self.teacher(global_crops)
            teacher_logits = teacher_logits.view(2, batch, -1).float()
            teacher_probabilities = self.center_loss.softmax_center_teacher(
                teacher_logits,
                teacher_temp=self.teacher_temperature(progress),
            )
            self.center_loss.update_center(teacher_logits.flatten(0, 1))
        global_features, student_global = self.student(global_crops)
        student_global = student_global.view(2, batch, -1)
        _, student_local = self.student(local_crops)
        student_local = student_local.view(8, batch, -1)
        with torch.autocast("cuda", enabled=False):
            # The user-supplied DINOv3 implementation owns the DINO loss.
            # Ten student crops x two teacher crops minus the two same-view
            # global pairs gives the selected 18 cross-view terms.
            dino = self.center_loss(
                torch.cat((student_global, student_local)).float(),
                teacher_probabilities.float(),
                ignore_diagonal=True,
            )
            features = global_features.float().view(2, batch, -1)
            koleo = torch.stack(
                [self.koleo_loss(view) for view in features]
            ).mean()
            loss = dino + KOLEO_WEIGHT * koleo
        return {
            "loss": loss,
            "dino": dino.detach(),
            "koleo": koleo.detach(),
        }
