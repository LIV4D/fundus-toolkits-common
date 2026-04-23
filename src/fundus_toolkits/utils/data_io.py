from datetime import datetime
from pathlib import Path
from typing import Optional
from .typing import PathLike

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "tif", "tiff", "bmp"]


def most_common_image_ext(
    directory: PathLike, raise_if_not_found: bool = True, extensions: list[str] | None = None
) -> str:
    from collections import Counter

    directory = Path(directory)
    exts_count = Counter(path.suffix for path in directory.glob("*"))

    if extensions is None:
        extensions = IMAGE_EXTENSIONS

    for ext, _ in exts_count.most_common():
        if ext[1:].lower() in extensions:
            return ext
    if raise_if_not_found:
        raise ValueError(f"No image extension found in directory {directory}")
    return ""


def overwrite_or_newer(src: Path, dst: Path, overwrite: Optional[bool | datetime] = None) -> bool:
    """Check if the destination file should be overwritten based on the source file's modification time and the overwrite flag.

    Parameters
    ----------
    src : Path
        The source file path.

    dst : Path
        The destination file path.

    overwrite : Optional[bool | datetime]
        If True, always overwrite the destination file. If False, never overwrite. If None, overwrite only if the source file is newer than the destination file. If a datetime is provided, overwrite only if the source file is newer than the specified datetime. Default is None.

    Returns
    -------
    bool
        True if the destination file should be overwritten, False otherwise.
    """  # noqa: E501
    if not dst.exists():
        return True
    if overwrite is None:
        return src.stat().st_mtime > dst.stat().st_mtime
    if isinstance(overwrite, datetime):
        return src.stat().st_mtime > overwrite.timestamp()
    return overwrite
