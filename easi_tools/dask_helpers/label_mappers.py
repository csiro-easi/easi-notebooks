import numpy as np


class LabelMapper:
    def map(self, raw_labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class WorldCoverLabelMapper(LabelMapper):
    def __init__(self, num_classes=11, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.code_map = {
            10: 0, 20: 1, 30: 2, 40: 3, 50: 4,
            60: 5, 70: 6, 80: 7, 90: 8, 95: 9, 100: 10,
        }
        self._lut = None  # built lazily

    def make_lut(self, max_code: int = 255) -> np.ndarray:
        """
        LUT where lut[raw_code] -> class_idx, and default is ignore_index.
        max_code=255 works for WorldCover codes; increase if needed.
        """
        lut = np.full((max_code + 1,), self.ignore_index, dtype=np.int64)
        for wc_code, cls_idx in self.code_map.items():
            if 0 <= wc_code <= max_code:
                lut[wc_code] = cls_idx
        return lut

    def map(self, raw_labels: np.ndarray) -> np.ndarray:
        # Handle NaNs safely by masking first (raw_labels may be float)
        y = np.asarray(raw_labels)
        nan_mask = np.isnan(y) if np.issubdtype(y.dtype, np.floating) else None

        # Convert to integer codes for LUT indexing
        y_int = y.astype(np.int64, copy=False)

        if self._lut is None:
            # LUT must cover max label value you expect
            # For safety, build to max(y_int) when feasible
            max_code = int(y_int.max()) if y_int.size else 255
            max_code = max(max_code, 255)
            self._lut = self.make_lut(max_code=max_code)

        # If y_int contains codes larger than LUT, rebuild once
        if y_int.size and int(y_int.max()) >= self._lut.shape[0]:
            self._lut = self.make_lut(max_code=int(y_int.max()))

        y_mapped = self._lut[y_int]

        if nan_mask is not None:
            y_mapped = y_mapped.copy()
            y_mapped[nan_mask] = self.ignore_index

        return y_mapped.astype(np.int64, copy=False)



class DictLUTLabelMapper(LabelMapper):
    def __init__(self, code_map: dict, ignore_index=255, max_code=255):
        self.code_map = dict(code_map)
        self.ignore_index = ignore_index
        self._lut = np.full((max_code + 1,), ignore_index, dtype=np.int64)
        for raw_code, cls_idx in self.code_map.items():
            if 0 <= int(raw_code) <= max_code:
                self._lut[int(raw_code)] = int(cls_idx)

    def map(self, raw_labels: np.ndarray) -> np.ndarray:
        y = np.asarray(raw_labels)
        nan_mask = np.isnan(y) if np.issubdtype(y.dtype, np.floating) else None
        y_int = y.astype(np.int64, copy=False)

        if y_int.size and int(y_int.max()) >= self._lut.shape[0]:
            # extend LUT if needed
            new_max = int(y_int.max())
            lut2 = np.full((new_max + 1,), self.ignore_index, dtype=np.int64)
            lut2[: self._lut.shape[0]] = self._lut
            self._lut = lut2

        out = self._lut[y_int]
        if nan_mask is not None:
            out = out.copy()
            out[nan_mask] = self.ignore_index
        return out