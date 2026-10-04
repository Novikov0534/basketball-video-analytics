"""Автоматическое разделение игроков на две команды.

Кроп формы преобразуется в эмбеддинг, затем признаки понижаются PCA или UMAP
и кластеризуются K-means. Команда трека определяется голосованием по кадрам.
"""

from collections import Counter, deque

import cv2
import numpy as np

SIGLIP_MODEL = "google/siglip-base-patch16-224"


def central_crop(frame, box, factor=0.4):
    """Центр рамки (40 % по ширине и высоте): майка без фона и соседей."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * factor / 2, (y2 - y1) * factor / 2
    height, width = frame.shape[:2]
    a, b = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    c, d = int(min(width, cx + half_w)), int(min(height, cy + half_h))
    if c - a < 4 or d - b < 4:
        return None
    return frame[b:d, a:c]


class ColorEmbedder:
    """Гистограмма LAB 4×6×6 + средний цвет. Быстро, без нейросети."""

    name = "color"

    def __call__(self, crops):
        features = []
        for crop in crops:
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            hist = cv2.calcHist([lab], [0, 1, 2], None, [4, 6, 6], [0, 256, 0, 256, 0, 256]).ravel()
            hist /= max(1.0, hist.sum())
            mean = lab.reshape(-1, 3).mean(axis=0) / 255.0
            features.append(np.concatenate([hist, mean]))
        return np.asarray(features, np.float32)


class SiglipEmbedder:
    """Эмбеддинги SigLIP (усреднение last_hidden_state, как у Roboflow)."""

    name = "siglip"

    def __init__(self, device="auto", batch_size=32):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL)
        self.model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL).to(device).eval()

    def __call__(self, crops):
        output = []
        with self.torch.no_grad():
            for start in range(0, len(crops), self.batch_size):
                batch = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops[start : start + self.batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                output.append(hidden.mean(dim=1).cpu().numpy())
        return np.concatenate(output).astype(np.float32)


class HybridEmbedder:
    """SigLIP плюс цвет формы.

    На одних эмбеддингах SigLIP кластеризация иногда цепляется не за форму, а
    за позу, фон или освещение (Roboflow обходит это через UMAP). Цветовая
    гистограмма добавляет решающий признак для светлой и тёмной формы, и обе
    части нормируются, чтобы ни одна не перевесила другую случайно.
    """

    name = "siglip+color"

    def __init__(self, siglip, color_weight=2.0):
        self.siglip, self.color, self.color_weight = siglip, ColorEmbedder(), color_weight

    def __call__(self, crops):
        def unit(features):
            return features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-9)

        return np.concatenate(
            [unit(self.siglip(crops)), unit(self.color(crops)) * self.color_weight], axis=1
        ).astype(np.float32)


def make_embedder(kind="auto", device="auto"):
    """auto: SigLIP + цвет, если SigLIP грузится, иначе только цвет."""
    if kind == "color":
        return ColorEmbedder(), None
    try:
        return HybridEmbedder(SiglipEmbedder(device)), None
    except Exception as exc:  # нет transformers, нет сети до Hugging Face и т. п.
        if kind == "siglip":
            raise
        return ColorEmbedder(), f"SigLIP недоступен ({type(exc).__name__}); команды разделены по цвету формы."


class TeamClusterer:
    """Понижение размерности + K-means (k=2) поверх эмбеддингов кропов.

    reducer="pca" — метод главных компонент: детерминирован и не требует
    дополнительных пакетов. reducer="umap" — альтернативное нелинейное понижение размерности.
    """

    def __init__(self, embedder, components=8, seed=42, reducer="pca"):
        self.embedder = embedder
        self.components = components
        self.seed = seed
        self.reducer_name = reducer
        self.reducer = None
        self.mean = self.basis = self.centers = None

    @property
    def fitted(self):
        return self.centers is not None

    def _project(self, features):
        if self.reducer is not None:
            return self.reducer.transform(features)
        return (features - self.mean) @ self.basis

    def _fit_reducer(self, features):
        if self.reducer_name == "umap":
            try:
                import umap
            except ImportError:
                raise RuntimeError(
                    "UMAP не установлен. Запустите scripts/bootstrap.py заново или установите umap-learn."
                ) from None

            self.reducer = umap.UMAP(n_components=3, random_state=self.seed)
            return self.reducer.fit_transform(features)
        self.mean = features.mean(axis=0)
        centered = features - self.mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.basis = vt[: min(self.components, vt.shape[0])].T
        return centered @ self.basis

    def fit(self, crops):
        if len(crops) < 8:
            return False
        features = self.embedder(crops)
        projected = np.float32(self._fit_reducer(features))
        cv2.setRNGSeed(self.seed)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
        _, labels, centers = cv2.kmeans(projected, 2, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        if min(np.bincount(labels, minlength=2)) < 3:  # один кластер почти пуст — разделения нет
            self.centers = None
            return False
        self.centers = centers
        return True

    def predict(self, crops):
        """→ массив (кластер, уверенность) для каждого кропа."""
        if not crops or not self.fitted:
            return []
        projected = self._project(self.embedder(crops))
        distances = np.linalg.norm(projected[:, None, :] - self.centers[None], axis=2)
        labels = distances.argmin(axis=1)
        # уверенность: насколько ближайший центр ближе второго (0 — посередине)
        margin = np.abs(distances[:, 0] - distances[:, 1]) / (distances.sum(axis=1) + 1e-9)
        return list(zip(labels.tolist(), margin.tolist()))


class TrackTeamVoter:
    """Команда трека = большинство голосов по последним кадрам.

    Строгое требование (min_votes голосов) действует при показе в реальном
    времени. В конце обработки, когда статистика уже собрана, применяется
    ослабленное правило: трек, проживший меньше времени, чем нужно для
    набора голосов, всё равно получает команду, если голоса единодушны.
    Без этого короткий трек — например, созданный после переподачи промптов
    SAM2 — остаётся без команды, и его очки повисают «ничьими».
    """

    def __init__(self, window=40, min_votes=3, min_share=0.6, min_margin=0.05):
        self.votes = {}
        self.window, self.min_votes, self.min_share, self.min_margin = window, min_votes, min_share, min_margin

    def add(self, track_id, label, margin):
        if margin >= self.min_margin:
            self.votes.setdefault(track_id, deque(maxlen=self.window)).append(int(label))

    def team(self, track_id, relaxed=False):
        votes = self.votes.get(track_id)
        if not votes:
            return None
        minimum = 1 if relaxed else self.min_votes
        share = 0.5 if relaxed else self.min_share
        if len(votes) < minimum:
            return None
        label, count = Counter(votes).most_common(1)[0]
        return label if count / len(votes) >= share else None


def collect_crops(frame, players, limit=None):
    crops = []
    for player in players:
        crop = central_crop(frame, player.box)
        if crop is not None:
            crops.append(crop)
    return crops[:limit] if limit else crops
