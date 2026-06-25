from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(ValueError):
    pass


def decode_image(data: bytes, max_pixels: int) -> Image.Image:
    if not data:
        raise InvalidImageError("图片为空")
    try:
        image = Image.open(BytesIO(data))
        width, height = image.size
        if width < 32 or height < 32:
            raise InvalidImageError("图片边长不能小于 32 像素")
        if width * height > max_pixels:
            raise InvalidImageError(f"图片像素数超过限制 {max_pixels}")
        image = ImageOps.exif_transpose(image)
        image.load()
        image = image.convert("RGB")
        image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
        return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("文件不是有效图片") from exc


def query_views(image: Image.Image) -> list[Image.Image]:
    """Full frame plus a conservative center crop for handheld photos."""
    width, height = image.size
    margin_x, margin_y = int(width * 0.06), int(height * 0.06)
    if margin_x < 2 or margin_y < 2:
        return [image]
    return [image, image.crop((margin_x, margin_y, width - margin_x, height - margin_y))]
