import asyncio
import logging
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

class ROISyncManager:
    """Gerencia sincronização periódica de ROIs com o painel ViewSense."""

    def __init__(self, api_client, config: dict):
        """
        Args:
            api_client: Instância do ViewSenseClient (com método sync_rois())
            config: Dict do config.yaml completo
        """
        self.api_client = api_client
        self.config = config
        self.current_rois: dict = {}          # {camera_id: [roi_configs]}
        self.current_cameras: dict = {}       # {camera_id: camera_config}
        
        # Default to 60s if not in config
        viewsense_conf = config.get("viewsense", {})
        self.sync_interval: int = viewsense_conf.get("roi_sync_interval_seconds", 60)
        
        self._running = False
        self._callbacks = []  # Funções chamadas quando ROIs mudam

    def on_roi_change(self, callback):
        """Registra callback chamado quando ROIs ou Configs de Câmera são atualizadas.
        
        O callback recebe: callback(camera_id: str, rois: list[dict], camera_config: dict)
        """
        self._callbacks.append(callback)

    async def start(self):
        """Inicia o loop de sincronização."""
        if self._running:
            return
            
        self._running = True
        logger.info(f"ROI Sync iniciado (intervalo: {self.sync_interval}s)")
        
        # Primeira sync imediata
        await self._sync()
        
        while self._running:
            await asyncio.sleep(self.sync_interval)
            await self._sync()

    def stop(self):
        """Para o loop de sincronização."""
        self._running = False
        logger.info("ROI Sync parado")

    async def _sync(self):
        """Executa uma sincronização."""
        try:
            result = await self.api_client.sync_rois()
            if result is None:
                # Pode acontecer se URLs estiverem vazias
                return

            if "error" in result:
                logger.error(f"Erro no ROI sync: {result['error']}")
                return

            new_rois = result.get("rois", [])
            new_cameras = result.get("cameras", [])
            pending_update = result.get("pending_update")
            
            # Agrupar ROIs por camera_id
            rois_by_camera: dict = {}
            for roi in new_rois:
                cam_id = roi["camera_id"]
                if cam_id not in rois_by_camera:
                    rois_by_camera[cam_id] = []
                rois_by_camera[cam_id].append(roi)

            # Map camera configs
            new_cameras_map = {c["id"]: c for c in new_cameras}
            
            # Detectar mudanças
            changed_cameras = self._detect_changes(rois_by_camera, new_cameras_map)
            
            # Atualizar estado interno
            self.current_rois = rois_by_camera
            self.current_cameras = new_cameras_map

            if changed_cameras:
                logger.info(f"ROIs/Configs atualizadas para câmeras: {changed_cameras}")
                for cam_id in changed_cameras:
                    rois = rois_by_camera.get(cam_id, [])
                    cam_config = new_cameras_map.get(cam_id, {})
                    
                    for cb in self._callbacks:
                        try:
                            cb(cam_id, rois, cam_config)
                        except Exception as e:
                            logger.error(f"Erro em callback ROI: {e}")
            
            # Tratar atualização de versão pendente (log apenas por enquanto)
            if pending_update:
                logger.warning(
                    f"⚠️ Atualização pendente: v{pending_update.get('target_version')} "
                    f"— {pending_update.get('update_notes', 'sem notas')}"
                )

        except Exception as e:
            logger.error(f"Falha no ROI sync: {e}")

    def _detect_changes(self, new_rois_by_camera: dict, new_cameras_map: dict) -> list:
        """Compara ROIs e Configs atuais com novas."""
        changed = []
        
        all_cameras = set(list(self.current_rois.keys()) + list(new_rois_by_camera.keys()) + 
                          list(self.current_cameras.keys()) + list(new_cameras_map.keys()))
        
        for cam_id in all_cameras:
            # Check ROI changes
            old_rois = self.current_rois.get(cam_id, [])
            new_rois = new_rois_by_camera.get(cam_id, [])
            
            rois_changed = False
            
            # Compare IDs and timestamps/versions if available, else content
            # Simplest for now: string representation or exact content compare
            if len(old_rois) != len(new_rois):
                rois_changed = True
            else:
                # Deep compare coordinates & metadata
                # Sort by ID to align
                old_rois.sort(key=lambda x: x.get('id', ''))
                new_rois.sort(key=lambda x: x.get('id', ''))
                
                if old_rois != new_rois:
                    rois_changed = True
            
            # Check Camera Config changes
            old_conf = self.current_cameras.get(cam_id, {})
            new_conf = new_cameras_map.get(cam_id, {})
            
            # Check specific fields that affect detection
            relevant_fields = ['confidence_threshold', 'enabled_classes', 'fps']
            
            # If camera config is completely new or missing
            if not old_conf and new_conf:
                rois_changed = True
            elif old_conf and not new_conf:
                pass # Camera removed - handled?
            else:
                 for field in relevant_fields:
                    if old_conf.get(field) != new_conf.get(field):
                        rois_changed = True
                        break

            if rois_changed:
                changed.append(cam_id)

        return changed

    def get_counting_lines(self, camera_id: str) -> list:
        """Retorna apenas linhas de contagem ativas para uma câmera."""
        rois = self.current_rois.get(camera_id, [])
        return [
            roi for roi in rois
            if roi.get("is_counting_line") and roi.get("is_active")
        ]

    def get_camera_config(self, camera_id: str) -> Optional[dict]:
        return self.current_cameras.get(camera_id)
