
"""
Módulo de detecção de cruzamento de linha.
Usa o centróide inferior (bottom-center) do bounding box do tracker
para determinar se um objeto cruzou uma linha de contagem.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

@dataclass
class CountingLine:
    """Representa uma linha de contagem vinda do ROI sync."""
    roi_id: str
    camera_id: str
    name: str
    # Pontos da linha em pixels (convertidos das coordenadas normalizadas)
    p1: Tuple[float, float]  # (x1, y1)
    p2: Tuple[float, float]  # (x2, y2)
    direction: str  # "both", "in", "out"
    
    # Vetor normal da linha (para determinar lado)
    _normal: Tuple[float, float] = field(init=False, repr=False)
    
    def __post_init__(self):
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        # Normal perpendicular (rotação 90° anti-horário)
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            self._normal = (-dy / length, dx / length)
        else:
            self._normal = (0, 1)
    
    def side_of_point(self, point: Tuple[float, float]) -> float:
        """
        Retorna valor positivo ou negativo indicando de qual lado
        da linha o ponto está. Sinal muda = cruzou a linha.
        """
        dx = point[0] - self.p1[0]
        dy = point[1] - self.p1[1]
        return dx * self._normal[0] + dy * self._normal[1]
    
    @classmethod
    def from_roi(cls, roi: dict, frame_width: int, frame_height: int) -> Optional["CountingLine"]:
        """Cria CountingLine a partir de uma ROI do roi-sync."""
        if roi.get("roi_type") != "line" or not roi.get("is_counting_line"):
            return None
        
        coords = roi.get("coordinates", [])
        if len(coords) < 2:
            return None
        
        # Ensure coordinates are float and normalized 0-1 before multiplying
        # Ideally ROI sync provides normalized coords.
        
        x1 = float(coords[0].get("x", 0))
        y1 = float(coords[0].get("y", 0))
        x2 = float(coords[1].get("x", 0))
        y2 = float(coords[1].get("y", 0))

        p1 = (x1 * frame_width, y1 * frame_height)
        p2 = (x2 * frame_width, y2 * frame_height)
        
        return cls(
            roi_id=roi["id"],
            camera_id=roi["camera_id"],
            name=roi.get("name", "Unnamed Line"),
            p1=p1,
            p2=p2,
            direction=roi.get("direction", "both"),
        )


class LineCrossingDetector:
    """
    Detecta quando tracks cruzam linhas de contagem.
    Mantém histórico do 'lado' de cada track para cada linha.
    """
    
    def __init__(self):
        # {(track_id, roi_id): último_side}
        self._track_sides: Dict[Tuple[str, str], float] = {}
        # Tracks já contados para evitar dupla contagem por linha
        # {(track_id, roi_id): direction}
        self._counted: Dict[Tuple[str, str], str] = {}
    
    def update(
        self,
        track_id: str,
        bbox: dict,  # {"x": int, "y": int, "width": int, "height": int}
        lines: List[CountingLine],
    ) -> List[dict]:
        """
        Verifica se o track cruzou alguma linha.
        
        Args:
            track_id: ID do tracker (ex: "track_42")
            bbox: Bounding box com x, y, width, height em pixels
            lines: Lista de CountingLine ativas
        
        Returns:
            Lista de eventos de cruzamento:
            [{"roi_id": str, "direction": "in"|"out", "crossed_line": True}]
        """
        if not lines:
            return []

        # Centróide inferior (bottom-center) - melhor para pedestres
        cx = bbox["x"] + bbox["width"] / 2
        cy = bbox["y"] + bbox["height"]  # bottom
        point = (cx, cy)
        
        crossings = []
        
        for line in lines:
            key = (track_id, line.roi_id)
            
            # Já foi contado nesta linha? Pula
            if key in self._counted:
                continue
            
            current_side = line.side_of_point(point)
            
            if key in self._track_sides:
                prev_side = self._track_sides[key]
                
                # Houve cruzamento? (sinais opostos)
                if prev_side * current_side < 0:
                    # Determinar direção: 
                    # positivo→negativo = "in", negativo→positivo = "out"
                    # Nota: Isso depende do sentido do vetor normal.
                    # Se normal aponta para "fora", então positivo é fora e negativo é dentro.
                    # Cruzar de + para - significa entrar (In).
                    
                    direction = "in" if prev_side > 0 else "out"
                    
                    # Filtrar por direção configurada na linha
                    if line.direction == "both" or line.direction == direction:
                        crossings.append({
                            "roi_id": line.roi_id,
                            "direction": direction,
                            "crossed_line": True,
                        })
                        self._counted[key] = direction
                    else:
                        # Log ignora para debug
                        pass # Logger is not available inside this class easily unless passed or global.
                        # We will let it pass silently here to keep class pure, 
                        # but in detector.py we capture the output.
                        # Wait, we can't capture it if we don't return it.
                        
                        # Better strategy: Return it with "crossed_line": False or a flag?
                        # No, that breaks contract.
                        # Let's import logging here.
                        import logging
                        logger = logging.getLogger("line_crossing")
                        logger.info(f"IGNORED CROSSING: {track_id} -> {direction} on line {line.name} (Config: {line.direction})")
            
            self._track_sides[key] = current_side
        
        return crossings
    
    def cleanup_stale_tracks(self, active_track_ids: set):
        """Remove tracks que não estão mais ativos (saíram do frame)."""
        stale_keys = [
            k for k in self._track_sides.keys()
            if k[0] not in active_track_ids
        ]
        for k in stale_keys:
            del self._track_sides[k]
            self._counted.pop(k, None)
