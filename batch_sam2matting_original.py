#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam2.build_sam import build_sam2matting
from sam2.sam2matting_image_predictor import SAM2MattingImagePredictor


# ============================================================
# SAM2Matting Tiny
# ============================================================

MODEL_CONFIG = "configs/sam2matting-sam2.1tiny.yaml"

MODEL_CHECKPOINT = (
    "checkpoints/SAM2Matting-SAM2.1Tiny.pt"
)


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

MASK_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# 文件工具
# ============================================================

def natural_key(path: Path) -> List[object]:
    """
    自然排序。

    例如：
        001.jpg
        002.jpg
        010.jpg
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_images(folder: Path) -> List[Path]:
    """
    获取输入 RGB 图片。
    """
    files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]

    return sorted(
        files,
        key=natural_key,
    )


def build_mask_index(folder: Path) -> Dict[str, Path]:
    """
    根据 stem 建立 mask 索引。

    例如：
        images/001.jpg
        masks/001.png

    会自动对应。
    """
    files = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in MASK_SUFFIXES
    ]

    files = sorted(
        files,
        key=lambda path: (
            natural_key(path),
            0 if path.suffix.lower() == ".png" else 1,
        ),
    )

    index = {}

    for path in files:
        if path.stem not in index:
            index[path.stem] = path
        else:
            print(
                f"[WARN] duplicate mask stem={path.stem}: "
                f"keep={index[path.stem].name}, "
                f"ignore={path.name}"
            )

    return index


# ============================================================
# Mask 处理
# ============================================================

def load_binary_mask(
    mask_path: Path,
    image_size,
    threshold: int,
):
    """
    读取 SAM2 粗 mask。

    最终转换成 bool：
        False = background
        True  = foreground
    """
    mask_image = Image.open(
        mask_path
    ).convert("L")

    if mask_image.size != image_size:
        old_size = mask_image.size

        mask_image = mask_image.resize(
            image_size,
            Image.Resampling.NEAREST,
        )

        print(
            f"       [WARN] mask resize: "
            f"{old_size} -> {image_size}"
        )

    mask_np = np.asarray(
        mask_image,
        dtype=np.uint8,
    )

    return mask_np >= threshold


def prepare_model_inputs(
    mask_binary: np.ndarray,
):
    """
    按 SAM2Matting 原始方式构造：

        raw_mask
        mask_input
    """
    mask_tensor = torch.from_numpy(
        mask_binary.copy()
    )

    raw_mask = mask_tensor.bool()

    # foreground = +10
    # background = -10
    mask_input = (
        mask_tensor.float() * 20.0 - 10.0
    )

    mask_input = (
        mask_input
        .unsqueeze(0)
        .unsqueeze(0)
    )

    mask_input = F.interpolate(
        mask_input,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )

    return raw_mask, mask_input


# ============================================================
# Alpha 处理
# ============================================================

def normalize_alpha(
    alpha,
    image_size,
):
    """
    将 SAM2Matting 输出转换为：

        H × W
        float32
        [0,1]
    """
    alpha_np = np.asarray(
        alpha,
        dtype=np.float32,
    )

    alpha_np = np.squeeze(
        alpha_np
    )

    if alpha_np.ndim != 2:
        raise RuntimeError(
            f"Unexpected alpha shape: "
            f"{alpha_np.shape}"
        )

    target_width = image_size[0]
    target_height = image_size[1]

    if alpha_np.shape != (
        target_height,
        target_width,
    ):
        print(
            f"       [WARN] alpha resize: "
            f"{alpha_np.shape} -> "
            f"{(target_height, target_width)}"
        )

        alpha_np = cv2.resize(
            alpha_np,
            (
                target_width,
                target_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    alpha_np = np.nan_to_num(
        alpha_np,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    alpha_np = np.clip(
        alpha_np,
        0.0,
        1.0,
    )

    return alpha_np


# ============================================================
# 输出
# ============================================================

def save_result(
    image: Image.Image,
    alpha: np.ndarray,
    stem: str,
    output_images_dir: Path,
    output_masks_dir: Path,
    jpeg_quality: int,
):
    """
    输出：

    output/images/<stem>.jpg
        黑背景 RGB。

    output/masks/<stem>.png
        连续 alpha。
    """
    rgb = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # alpha PNG
    # --------------------------------------------------------

    alpha_u8 = np.clip(
        np.rint(alpha * 255.0),
        0,
        255,
    ).astype(np.uint8)

    alpha_path = (
        output_masks_dir
        / f"{stem}.png"
    )

    Image.fromarray(
        alpha_u8,
        mode="L",
    ).save(
        alpha_path
    )

    # --------------------------------------------------------
    # Black-background RGB
    # --------------------------------------------------------

    alpha_3 = alpha[..., None]

    black = (
        rgb.astype(np.float32)
        * alpha_3
    )

    black = np.clip(
        np.rint(black),
        0,
        255,
    ).astype(np.uint8)

    black_path = (
        output_images_dir
        / f"{stem}.jpg"
    )

    Image.fromarray(
        black,
        mode="RGB",
    ).save(
        black_path,
        format="JPEG",
        quality=jpeg_quality,
        subsampling=0,
        optimize=True,
    )

    print(
        f"       [OK] "
        f"alpha=[{alpha_u8.min()}, {alpha_u8.max()}], "
        f"unique={len(np.unique(alpha_u8))}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Batch SAM2Matting inference. "
            "Input original images + SAM2 coarse masks; "
            "output black-background images and alpha masks."
        )
    )

    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help=(
            "原始 RGB 图片目录。"
        ),
    )

    parser.add_argument(
        "--masks",
        type=Path,
        required=True,
        help=(
            "SAM2 粗二值 mask 目录。"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "输出根目录。最终生成 "
            "output/images 和 output/masks。"
        ),
    )

    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=128,
        help=(
            "粗 mask 二值化阈值。"
            "默认 128。"
        ),
    )

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help=(
            "黑背景 JPG 保存质量。"
            "默认 95。"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "只处理前 N 张图片。"
            "0 表示全部。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "覆盖已经存在的输出。"
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    repo_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    os.chdir(repo_root)

    image_dir = (
        args.images
        .expanduser()
        .resolve()
    )

    mask_dir = (
        args.masks
        .expanduser()
        .resolve()
    )

    output_root = (
        args.output
        .expanduser()
        .resolve()
    )

    output_images_dir = (
        output_root / "images"
    )

    output_masks_dir = (
        output_root / "masks"
    )

    # --------------------------------------------------------
    # 参数检查
    # --------------------------------------------------------

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: "
            f"{image_dir}"
        )

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Masks directory not found: "
            f"{mask_dir}"
        )

    if not (
        0 <= args.mask_threshold <= 255
    ):
        raise ValueError(
            "--mask_threshold must be in [0,255]"
        )

    if not (
        1 <= args.jpeg_quality <= 100
    ):
        raise ValueError(
            "--jpeg_quality must be in [1,100]"
        )

    if args.limit < 0:
        raise ValueError(
            "--limit must be >= 0"
        )

    checkpoint_path = (
        repo_root / MODEL_CHECKPOINT
    ).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    output_images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_masks_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 数据
    # --------------------------------------------------------

    images = list_images(
        image_dir
    )

    mask_index = build_mask_index(
        mask_dir
    )

    if not images:
        raise RuntimeError(
            f"No images found in: "
            f"{image_dir}"
        )

    if not mask_index:
        raise RuntimeError(
            f"No masks found in: "
            f"{mask_dir}"
        )

    if args.limit > 0:
        images = images[
            :args.limit
        ]

    # --------------------------------------------------------
    # 信息
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "SAM2Matting Batch Interface"
    )
    print("=" * 72)

    print(
        f"Input images : "
        f"{image_dir}"
    )

    print(
        f"Input masks  : "
        f"{mask_dir}"
    )

    print(
        f"Output       : "
        f"{output_root}"
    )

    print(
        f"Output images: "
        f"{output_images_dir}"
    )

    print(
        f"Output masks : "
        f"{output_masks_dir}"
    )

    print(
        f"Image count  : "
        f"{len(images)}"
    )

    print(
        f"Mask count   : "
        f"{len(mask_index)}"
    )

    print(
        f"Model        : "
        f"SAM2Matting-SAM2.1Tiny"
    )

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print("=" * 72)
    print()

    # --------------------------------------------------------
    # 模型只加载一次
    # --------------------------------------------------------

    print(
        "[INFO] Loading SAM2Matting Tiny..."
    )

    model = build_sam2matting(
        MODEL_CONFIG,
        str(checkpoint_path),
    )

    predictor = (
        SAM2MattingImagePredictor(
            model
        )
    )

    print(
        "[INFO] Model loaded."
    )

    print()

    # --------------------------------------------------------
    # 批量推理
    # --------------------------------------------------------

    processed = 0
    skipped = 0
    missing_mask = 0
    failed = 0

    total = len(images)

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )
    ):

        for idx, image_path in enumerate(
            images,
            start=1,
        ):

            stem = image_path.stem

            print(
                f"[{idx:04d}/{total:04d}] "
                f"{image_path.name}"
            )

            mask_path = (
                mask_index.get(stem)
            )

            if mask_path is None:

                print(
                    "       [SKIP] "
                    "missing corresponding mask"
                )

                missing_mask += 1
                continue

            output_image_path = (
                output_images_dir
                / f"{stem}.jpg"
            )

            output_mask_path = (
                output_masks_dir
                / f"{stem}.png"
            )

            if (
                not args.overwrite
                and output_image_path.is_file()
                and output_mask_path.is_file()
            ):

                print(
                    "       [SKIP] "
                    "output exists"
                )

                skipped += 1
                continue

            try:

                # --------------------------------------------
                # 原始 RGB
                # --------------------------------------------

                image = Image.open(
                    image_path
                ).convert("RGB")

                # --------------------------------------------
                # SAM2 coarse mask
                # --------------------------------------------

                mask_binary = (
                    load_binary_mask(
                        mask_path=mask_path,
                        image_size=image.size,
                        threshold=args.mask_threshold,
                    )
                )

                foreground_pixels = int(
                    np.count_nonzero(
                        mask_binary
                    )
                )

                if foreground_pixels == 0:

                    print(
                        "       [ERROR] "
                        "empty foreground mask"
                    )

                    failed += 1
                    continue

                foreground_ratio = (
                    foreground_pixels
                    / mask_binary.size
                )

                print(
                    f"       coarse mask: "
                    f"{foreground_ratio:.4%} foreground"
                )

                # --------------------------------------------
                # Prompt
                # --------------------------------------------

                raw_mask, mask_input = (
                    prepare_model_inputs(
                        mask_binary
                    )
                )

                # --------------------------------------------
                # SAM2 image encoder
                # --------------------------------------------

                encoded_image = (
                    predictor.set_image(
                        image
                    )
                )

                # --------------------------------------------
                # 原版 SAM2Matting
                # --------------------------------------------

                _, alpha, _ = (
                    predictor.predict(
                        img=encoded_image,
                        raw_mask=raw_mask,
                        mask_input=mask_input,
                        multimask_output=False,
                    )
                )

                # --------------------------------------------
                # Alpha
                # --------------------------------------------

                alpha = normalize_alpha(
                    alpha=alpha,
                    image_size=image.size,
                )

                # --------------------------------------------
                # 保存
                # --------------------------------------------

                save_result(
                    image=image,
                    alpha=alpha,
                    stem=stem,
                    output_images_dir=(
                        output_images_dir
                    ),
                    output_masks_dir=(
                        output_masks_dir
                    ),
                    jpeg_quality=(
                        args.jpeg_quality
                    ),
                )

                processed += 1

            except Exception as exc:

                failed += 1

                print(
                    f"       [ERROR] "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("Finished")
    print("=" * 72)

    print(
        f"Processed   : {processed}"
    )

    print(
        f"Skipped     : {skipped}"
    )

    print(
        f"Missing mask: {missing_mask}"
    )

    print(
        f"Failed      : {failed}"
    )

    print()

    print(
        f"Images: "
        f"{output_images_dir}"
    )

    print(
        f"Masks : "
        f"{output_masks_dir}"
    )

    print("=" * 72)

    if failed > 0:
        raise RuntimeError(
            f"{failed} image(s) failed."
        )


if __name__ == "__main__":
    main()
