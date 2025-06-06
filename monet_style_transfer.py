#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Arte con Estilo Monet
----------------------------------
Uso rápido (recomendado dentro de un venv o conda):

    pip install torch torchvision pillow matplotlib
    python monet_style_transfer.py \
        --content_path foto.jpg \
        --style_path monet.jpg \
        --output_path resultado.png

El script:

1. Carga y pre-procesa la imagen de contenido y la de estilo.
2. Carga un modelo VGG-19 preentrenado y define las pérdidas de contenido y estilo.
3. Optimiza la imagen generada durante varias iteraciones.
4. Cada 50 pasos guarda una vista previa del progreso (progress_0000.png, progress_0050.png, …).
5. Des-normaliza el tensor final y lo guarda en el archivo indicado (sin autocontrast).
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms

# ────────────────────────── Parámetros globales ──────────────────────────

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

IMSIZE = 512
STYLE_LAYERS   = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
CONTENT_LAYERS = ["conv4_2"]

STYLE_WEIGHT   = 1e6
CONTENT_WEIGHT = 1
TV_WEIGHT      = 1e-6
NUM_STEPS      = 400          # ← ahora 400
LR             = 0.03
SAVE_EVERY     = 50           # guardar progreso cada 50 pasos

# ──────────────────────────── Transformaciones ───────────────────────────

loader = transforms.Compose([
    transforms.Resize((IMSIZE, IMSIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

def undo_normalize(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    return t * std + mean

unloader = transforms.Compose([
    transforms.Lambda(undo_normalize),
    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
    transforms.ToPILImage(),
])

# ──────────────────────────── Utilidades ─────────────────────────────────

def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return loader(img).unsqueeze(0).to(DEVICE)

def save_tensor(t: torch.Tensor, path: Path) -> None:
    img = t.detach().cpu().squeeze(0)          # C, H, W
    pil_img = unloader(img)
    pil_img.save(path)
    print(f"✅ Imagen guardada en {path}")

def gram_matrix(t: torch.Tensor) -> torch.Tensor:
    b, c, h, w = t.size()
    f = t.view(b * c, h * w)
    return (f @ f.t()) / (b * c * h * w)

# ──────────────────────── Módulos de pérdida ─────────────────────────────

class StyleLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = gram_matrix(target).detach()
        self.loss   = torch.tensor(0.)

    def forward(self, x):
        self.loss = nn.functional.mse_loss(gram_matrix(x), self.target)
        return x

class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss   = torch.tensor(0.)

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.tensor(0.)

    def forward(self, x):
        self.loss = (
            torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        )
        return x

# ────────────────── Construcción del modelo NST ──────────────────────────

def build_model(
    cnn: nn.Module,
    style_img: torch.Tensor,
    content_img: torch.Tensor,
    style_layers: List[str],
    content_layers: List[str],
) -> Tuple[nn.Module, List[StyleLoss], List[ContentLoss], TVLoss]:

    cnn = cnn.features.to(DEVICE).eval()

    style_losses, content_losses = [], []
    tv_loss = TVLoss()

    model = nn.Sequential()
    conv_block, conv_in_block = 0, 0

    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            # Gestionar nombres tipo conv{block}_{index}
            if conv_in_block == 0:
                conv_block += 1
            conv_in_block += 1
            name = f"conv{conv_block}_{conv_in_block}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu{conv_block}_{conv_in_block}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool{conv_block}"
            layer = nn.AvgPool2d(kernel_size=2, stride=2)
            conv_in_block = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn{conv_block}_{conv_in_block}"
        else:
            raise RuntimeError(f"Capa no soportada: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            c_loss = ContentLoss(target)
            model.add_module(f"content_loss_{name}", c_loss)
            content_losses.append(c_loss)

        if name in style_layers:
            target = model(style_img).detach()
            s_loss = StyleLoss(target)
            model.add_module(f"style_loss_{name}", s_loss)
            style_losses.append(s_loss)

    model.add_module("tv_loss", tv_loss)
    return model, style_losses, content_losses, tv_loss

# ──────────────────────── Bucle de optimización ──────────────────────────

# ……………………… run_style_transfer ……………………………
def run_style_transfer(
    model, input_img,
    style_losses, content_losses, tv_loss,
    output_prefix: Path,
    num_steps=NUM_STEPS,
    save_every=SAVE_EVERY,
    style_w=STYLE_WEIGHT, content_w=CONTENT_WEIGHT, tv_w=TV_WEIGHT
):
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=LR)
    print(f"🖼️  Optimizando en {DEVICE}…")
    step = [0]
    loss_history = []

    save_tensor(input_img, output_prefix.with_name(f"{output_prefix.stem}_progress_{step[0]:04d}.png"))

    while step[0] < num_steps:

        def closure():
            optimizer.zero_grad()
            model(input_img)

            s_loss = sum(sl.loss for sl in style_losses)
            c_loss = sum(cl.loss for cl in content_losses)
            t_loss = tv_loss.loss

            loss = style_w * s_loss + content_w * c_loss + tv_w * t_loss
            loss.backward()

            if step[0] % save_every == 0:
                print(f"{step[0]:3d}/{num_steps}  "
                      f"Estilo: {s_loss.item():.4f}  "
                      f"Contenido: {c_loss.item():.4f}  "
                      f"TV: {t_loss.item():.4f}")
                save_tensor(
                    input_img,
                    output_prefix.with_name(f"{output_prefix.stem}_progress_{step[0]:04d}.png")
                )

            # ---- guarda pérdidas en la lista ----
            loss_history.append(loss.item())        # ← NUEVO

            step[0] += 1
            return loss

        optimizer.step(closure)

    # ➊ Devuelve imagen
    result = input_img.detach()

    # ➋ Genera la gráfica y guárdala
    plt.figure(figsize=(6,4))
    plt.plot(loss_history)
    plt.title("Curva de pérdida total")
    plt.xlabel("Iteración")
    plt.ylabel("Loss")
    curve_path = output_prefix.with_name(f"{output_prefix.stem}_loss_curve.png")
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    print(f"📈 Gráfica guardada en {curve_path}")

    return result

# ─────────────────────────────── Main ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer Monet")
    parser.add_argument("--content_path", type=Path, required=True)
    parser.add_argument("--style_path",   type=Path, required=True)
    parser.add_argument("--output_path",  type=Path, default="output.png")
    args = parser.parse_args()

    content_img = load_image(args.content_path)
    style_img   = load_image(args.style_path)

    input_img   = content_img.clone()

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    model, s_losses, c_losses, tv_loss = build_model(
        cnn, style_img, content_img, STYLE_LAYERS, CONTENT_LAYERS
    )

    print(f"Capas de estilo encontradas:   {len(s_losses)}")
    print(f"Capas de contenido encontradas: {len(c_losses)}")

    output = run_style_transfer(
        model, input_img,
        s_losses, c_losses, tv_loss,
        output_prefix=args.output_path,
    )

    # Guarda la versión final
    save_tensor(output, args.output_path)

if __name__ == "__main__":
    main()
