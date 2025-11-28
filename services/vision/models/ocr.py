import logging
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from paddleocr import PaddleOCR
from PIL import Image

warnings.filterwarnings(
    "ignore",
    message=".*No ccache found.*",
    module="paddle",
)
logging.getLogger("paddlex").setLevel(logging.ERROR)

_txt_det_path = Path(__file__).parent / "det"
_txt_rec_path = Path(__file__).parent / "rec"

if not _txt_det_path.exists() or not _txt_rec_path.exists():
    raise FileNotFoundError("OCR model files are missing.")


class OcrPrediction(TypedDict, total=False):
    """OCR prediction result."""

    dt_polys: NDArray[np.int32 | np.int16]
    text_type: str
    text_rec_score_thresh: float
    return_word_box: bool
    rec_texts: list[str]
    rec_scores: list[float]
    rec_polys: list[NDArray[np.int32 | np.int16]]
    rec_boxes: NDArray[np.int32 | np.int16]


class OcrResult(TypedDict):
    """Individual OCR result."""

    text: str
    score: float
    polygon: list[list[int]]
    box: list[int]


@dataclass(frozen=True, slots=True)
class Analysis:
    """OCR analysis result."""

    results: list[OcrResult]


class SingletonMeta(type):
    """Singleton pattern to guarantee a single class instance."""

    _instances: dict[type, object] = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """Returns the singleton instance of the class."""

        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

        return cls._instances[cls]


class TextExtractor(metaclass=SingletonMeta):
    """OCR predictor using PaddleOCR."""

    def __init__(self, **kwargs) -> None:
        """Initializes the OCR predictor with the given configuration."""

        self._ocr = PaddleOCR(
            text_detection_model_dir=str(_txt_det_path),
            text_recognition_model_dir=str(_txt_rec_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **kwargs,
        )
        # Lock to ensure thread-safe inference
        self._inference_lock = Lock()

    @property
    def model(self) -> PaddleOCR:
        """Returns the underlying PaddleOCR model."""

        return self._ocr

    def predict(self, data: bytes, **kwargs) -> OcrPrediction:
        """Performs OCR prediction on the input data."""

        img = Image.open(BytesIO(data)).convert("RGB")
        pixels = np.array(img)[:, :, ::-1]

        with self._inference_lock:
            return self._ocr.predict(input=pixels, **kwargs)[0]

    def __call__(self, data: bytes, **kwargs) -> Analysis:
        """Performs OCR prediction on the input data."""

        pred = self.predict(data, **kwargs)
        results: list[OcrResult] = []

        texts = pred["rec_texts"]
        scores: list[float] = pred["rec_scores"]
        boxes: list[list[int]] = pred["rec_boxes"].tolist()
        polygons: list[list[list[int]]] = [p.tolist() for p in pred["rec_polys"]]

        for t, s, b, p in zip(texts, scores, boxes, polygons):
            results.append(OcrResult(text=t, score=s, box=b, polygon=p))

        return Analysis(results=results)
