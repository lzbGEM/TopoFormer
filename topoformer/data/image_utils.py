import numpy as np
import cv2
from PIL import Image as PILImage

try:
    import tifffile  # type: ignore
except Exception:
    tifffile = None


def rgb_to_gray_np(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim != 3:
        raise ValueError(f"rgb_to_gray_np expects 3D, got {x.shape}")

    if x.shape[-1] in (3, 4):
        if x.shape[-1] == 4:
            x = x[..., :3]
        if x.dtype == np.float32 or x.dtype == np.float64:
            return x.mean(axis=-1).astype(np.float32)
        x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return x.astype(np.float32)

    if x.shape[0] in (3, 4):
        if x.shape[0] == 4:
            x = x[:3, ...]
        x = np.transpose(x, (1, 2, 0))
        if x.dtype == np.float32 or x.dtype == np.float64:
            return x.mean(axis=-1).astype(np.float32)
        x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return x.astype(np.float32)

    raise ValueError(f"Unexpected RGB shape: {x.shape}")


def normalize_image_to_01(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + eps)


def _to_float01_rgb(img: np.ndarray, *, allow_grayscale_to_rgb: bool = False) -> np.ndarray:
    x = np.asarray(img)

    if x.ndim == 2:
        if not allow_grayscale_to_rgb:
            raise ValueError(
                "Got grayscale image array with shape HxW. "
                "Set allow_grayscale_to_rgb=True if this conversion is intended."
            )
        x = np.stack([x, x, x], axis=-1)

    elif x.ndim == 3:
        if x.shape[0] in (3, 4) and x.shape[-1] not in (3, 4):
            x = np.transpose(x, (1, 2, 0))

        if x.shape[-1] == 4:
            x = x[..., :3]

        if x.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape for RGB: {x.shape}")

    else:
        raise ValueError(f"Unexpected image shape: {x.shape}")

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        denom = float(info.max) if info.max > 0 else 255.0
        x = x.astype(np.float32) / denom
    else:
        x = x.astype(np.float32)
        mx = float(np.nanmax(x)) if x.size else 0.0
        if mx > 1.5:
            if mx <= 255.0:
                x = x / 255.0
            elif mx <= 65535.0:
                x = x / 65535.0
            else:
                x = x / (mx if mx > 0 else 1.0)

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def to_01_1chw_224(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 3:
        x = rgb_to_gray_np(x)
    elif x.ndim != 2:
        raise ValueError(f"Unexpected image shape: {x.shape}")

    x01 = normalize_image_to_01(x)
    x224 = cv2.resize(x01, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    return x224[None, :, :]


def to_01_3chw_224(img: np.ndarray, *, allow_grayscale_to_rgb: bool = False) -> np.ndarray:
    x = _to_float01_rgb(img, allow_grayscale_to_rgb=allow_grayscale_to_rgb)
    x224 = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.transpose(x224, (2, 0, 1)).astype(np.float32)


def read_image_any(path: str) -> np.ndarray:
    p = str(path)
    pl = p.lower()

    if pl.endswith((".tif", ".tiff")):
        if tifffile is not None:
            arr = tifffile.imread(p)
            arr = np.asarray(arr)
            while arr.ndim > 3:
                arr = arr[0]
            return arr

        with PILImage.open(p) as im:
            return np.array(im)

    if pl.endswith((".png", ".jpg", ".jpeg")):
        with PILImage.open(p) as im:
            return np.array(im.convert("RGB"))

    raise ValueError(f"Unsupported image format: {path}")