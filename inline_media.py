import base64
import mimetypes
from pathlib import Path
from typing import Optional, Union


def file_to_data_url(path_like: Union[str, Path, None]) -> Optional[str]:
    if not path_like:
        return None

    path = Path(path_like)
    if not path.exists() or not path.is_file():
        return None

    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
