"""
Módulo de detecção de cruzamento de linha.
Suporta linhas retas (2 pontos) e polilinhas (N pontos → N-1 segmentos).

Ponto sensor adaptativo:
  - Movimento para CIMA  (cy ↓): usa topo  da bbox (cabeça chega primeiro)
  - Movimento para BAIXO (cy ↑): usa base  da bbox (pés chegam primeiro)
  - Movimento para ESQUERDA (cx ↓): usa borda esquerda
  - Movimento para DIREITA  (cx ↑): usa borda direita
  - Estático / sem histórico: usa centro

Deduplication: 1 evento por (track_id, roi_id) por CROSSING_COOLDOWN_S segundos.
"""
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)

# Cooldown: evita re-contagem se o track fica oscilando na linha
CROSSING_COOLDOWN_S = 4.0

# Limiar mínimo de movimento (px) para considerar que houve deslocamento
MOVEMENT_THRESHOLD = 2


@dataclass
class CountingLine:
    """
    Linha de contagem — reta (2 pontos) ou polilinha (N pontos).
    Internamente decomposta em N-1 segmentos, todos com o mesmo roi_id.
    """
    roi_id: str
    camera_id: str
    name: str
    points: List[Tuple[float, float]]   # [(x0,y0), ..., (xN,yN)]
    direction: str  # "both" | "in" | "out"

    _normals: List[Tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self):
        self._normals = []
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            dx, dy = x2 - x1, y2 - y1
            length = float(np.sqrt(dx ** 2 + dy ** 2))
            if length > 0:
                self._normals.append((-dy / length, dx / length))
            else:
                self._normals.append((0.0, 1.0))

    # Backwards-compat properties for 2-point access
    @property
    def p1(self) -> Tuple[float, float]:
        return self.points[0]

    @property
    def p2(self) -> Tuple[float, float]:
        return self.points[-1]

    def side_of_point(self, point: Tuple[float, float]) -> float:
        """Legacy helper — uses first segment normal (2-point lines)."""
        return self.side_of_point_for_segment(0, point)

    def side_of_point_for_segment(self, seg_idx: int, point: Tuple[float, float]) -> float:
        nx, ny = self._normals[seg_idx]
        ox, oy = self.points[seg_idx]
        return (point[0] - ox) * nx + (point[1] - oy) * ny

    @classmethod
    def from_roi(cls, roi: dict, frame_width: int, frame_height: int) -> Optional["CountingLine"]:
        """
        Cria CountingLine a partir de ROI do roi-sync.
        Aceita roi_type 'line' (2 pts) e 'polyline' (N pts).
        """
        roi_type = roi.get("roi_type", "")
        if roi_type not in ("line", "polyline"):
            return None
        if not roi.get("is_counting_line"):
            return None

        coords = roi.get("coordinates", [])
        if len(coords) < 2:
            return None

        points = []
        for c in coords:
            x = float(c.get("x", 0))
            y = float(c.get("y", 0))
            # Normalized 0-1 → pixels
            px = x * frame_width  if x <= 1.0 else x
            py = y * frame_height if y <= 1.0 else y
            points.append((px, py))

        return cls(
            roi_id=roi["id"],
            camera_id=roi["camera_id"],
            name=roi.get("name", "Linha"),
            points=points,
            direction=roi.get("direction", "both"),
        )


class LineCrossingDetector:
    """
    Detecta cruzamentos de tracks com linhas/polilinhas.
    Usa ponto sensor adaptativo (borda líder da bbox conforme direção de movimento).
    Emite no máximo 1 evento por (track_id, roi_id) por cooldown window.
    """

    def __init__(self):
        # {(track_id, roi_id, seg_idx): prev_side}
        self._track_sides: Dict[Tuple[str, str, int], float] = {}
        # {(track_id, roi_id): timestamp_of_last_crossing}
        self._counted: Dict[Tuple[str, str], float] = {}
        # {track_id: (prev_cx, prev_cy)} — histórico de posição central
        self._prev_center: Dict[str, Tuple[float, float]] = {}

    def _leading_point(self, bbox: dict) -> Tuple[float, float]:
        """
        Retorna o ponto sensor que chega primeiro à linha de contagem,
        de acordo com a direção de movimento da bbox.

        Coordenadas de imagem: Y cresce para baixo, X cresce para direita.
          - Movimento ↑ (cy ↓): topo da bbox (cabeça)
          - Movimento ↓ (cy ↑): base  da bbox (pés)
          - Movimento ← (cx ↓): borda esquerda
          - Movimento → (cx ↑): borda direita
          - Sem histórico / estático: centro
        """
        cx_center = bbox["x"] + bbox["width"]  / 2
        cy_center = bbox["y"] + bbox["height"] / 2

        prev = self._prev_center.get(str(id(bbox)))   # id trick no funciona
        # Usar track_id passado externamente — ver _update_center()
        return (cx_center, cy_center)   # fallback; sobreescrito em update()

    def _compute_sensor_point(
        self,
        track_id: str,
        bbox: dict,
    ) -> Tuple[float, float]:
        """
        Calcula o ponto sensor adaptativo para este track.
        Atualiza o histórico de posição.
        """
        cx_center = bbox["x"] + bbox["width"]  / 2
        cy_center = bbox["y"] + bbox["height"] / 2

        cx_left  = float(bbox["x"])
        cx_right = float(bbox["x"] + bbox["width"])
        cy_top   = float(bbox["y"])                    # cabeça (y menor)
        cy_bot   = float(bbox["y"] + bbox["height"])   # pés   (y maior)

        prev = self._prev_center.get(track_id)

        if prev is not None:
            prev_cx, prev_cy = prev

            # Vertical: movimento para cima (cy diminui) → usa cabeça
            if   prev_cy - cy_center > MOVEMENT_THRESHOLD:
                cy = cy_top
            # Vertical: movimento para baixo (cy aumenta) → usa pés
            elif cy_center - prev_cy > MOVEMENT_THRESHOLD:
                cy = cy_bot
            else:
                cy = cy_center

            # Horizontal: movimento para esquerda (cx diminui) → borda esquerda
            if   prev_cx - cx_center > MOVEMENT_THRESHOLD:
                cx = cx_left
            # Horizontal: movimento para direita (cx aumenta) → borda direita
            elif cx_center - prev_cx > MOVEMENT_THRESHOLD:
                cx = cx_right
            else:
                cx = cx_center
        else:
            cx, cy = cx_center, cy_center

        # Salva centro atual para próximo frame
        self._prev_center[track_id] = (cx_center, cy_center)
        return (cx, cy)

    def update(
        self,
        track_id: str,
        bbox: dict,
        lines: List[CountingLine],
    ) -> List[dict]:
        """
        Verifica se o track cruzou alguma linha ou polilinha.
        Retorna lista de eventos — máximo 1 por roi_id por chamada.
        """
        if not lines:
            return []

        point = self._compute_sensor_point(track_id, bbox)

        crossings = []
        now = time.time()
        fired_this_frame: set = set()

        for line in lines:
            roi_key = (track_id, line.roi_id)

            # Cooldown: skip if crossed recently
            if roi_key in self._counted:
                if now - self._counted[roi_key] < CROSSING_COOLDOWN_S:
                    # Garante que os lados são atualizados mesmo no cooldown
                    # (evita re-fire logo após o cooldown expirar com side stale)
                    n_segs = len(line.points) - 1
                    for seg_idx in range(n_segs):
                        seg_key = (track_id, line.roi_id, seg_idx)
                        self._track_sides[seg_key] = line.side_of_point_for_segment(seg_idx, point)
                    continue

            if line.roi_id in fired_this_frame:
                continue

            n_segs = len(line.points) - 1
            crossed = False

            for seg_idx in range(n_segs):
                seg_key = (track_id, line.roi_id, seg_idx)
                current_side = line.side_of_point_for_segment(seg_idx, point)

                if seg_key in self._track_sides:
                    prev_side = self._track_sides[seg_key]

                    if prev_side * current_side < 0:   # sinal mudou → cruzou
                        direction = "in" if prev_side > 0 else "out"

                        # IMPORTANTE: sempre atualiza o lado antes de qualquer break
                        self._track_sides[seg_key] = current_side

                        if line.direction == "both" or line.direction == direction:
                            seg_label = f" seg {seg_idx + 1}/{n_segs}" if n_segs > 1 else ""
                            logger.info(
                                f"CROSSING: {track_id} → {direction} "
                                f"'{line.name}'{seg_label} (roi={line.roi_id[:8]})"
                            )
                            crossings.append({
                                "roi_id": line.roi_id,
                                "direction": direction,
                                "crossed_line": True,
                            })
                            self._counted[roi_key] = now
                            fired_this_frame.add(line.roi_id)
                            crossed = True
                        else:
                            logger.debug(
                                f"IGNORED: {track_id} → {direction} on '{line.name}' "
                                f"(expected: {line.direction})"
                            )
                        break  # só processa um segmento por linha por frame
                    else:
                        # Mesmo lado — atualiza normalmente
                        self._track_sides[seg_key] = current_side
                else:
                    # Primeira vez vendo este segmento — registra sem cruzamento
                    self._track_sides[seg_key] = current_side

                if crossed:
                    break

        return crossings

    def cleanup_stale_tracks(self, active_track_ids: set):
        """Remove estado de tracks que saíram do frame."""
        for keys, store in [
            ([k for k in self._track_sides if k[0] not in active_track_ids], self._track_sides),
            ([k for k in self._counted if k[0] not in active_track_ids], self._counted),
        ]:
            for k in keys:
                store.pop(k, None)

        for tid in [t for t in self._prev_center if t not in active_track_ids]:
            del self._prev_center[tid]
