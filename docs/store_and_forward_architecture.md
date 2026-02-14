# Arquitetura de "Store & Forward" com SQLite (Offline-First)

## Descrição
Para aumentar a robustez da aplicação em cenários de internet instável e reduzir o tráfego de rede (chaves API, handshakes TLS), vamos implementar um sistema de buffer persistente usando **SQLite**.

Atualmente, o `api_client.py` mantém uma fila em memória (`self.queue`). Se o serviço reiniciar ou cair, os dados não enviados são perdidos.

## Objetivos
1.  **Persistência:** Nenhum dado de detecção deve ser perdido se o serviço reiniciar ou a internet cair.
2.  **Otimização de Tráfego:** Enviar dados em lotes maiores (ex: a cada 60s ou 100 registros), reduzindo overhead de conexão.
3.  **Resiliência:** O sistema deve continuar detectando e gravando localmente mesmo sem conexão com a API.

## Plano de Implementação

### 1. Novo Módulo de Banco de Dados (`database.py`)
Criar uma classe `LocalDB` responsável por gerenciar o arquivo `detections.db`.

```python
# Esboço do Schema
CREATE TABLE IF NOT EXISTS pending_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    payload JSON NOT NULL,  -- O JSON completo da detecção
    retry_count INTEGER DEFAULT 0
);
```

### 2. Alteração no Fluxo de Detecção (`detector.py` -> `database.py`)
Em vez de chamar `api_client.add_detection()`, o detector chamará `local_db.save_detection()`.
Isso deve ser **extremamente rápido** (INSERT no SQLite) para não bloquear o processamento de vídeo.

### 3. Worker de Sincronização em Segundo Plano (`sync_worker.py`)
Uma thread ou processo separado que roda em loop infinito:

1.  **Verificação:** A cada `sync_interval` (ex: 30s) ou quando `pending_count > batch_size`.
2.  **Leitura:** `SELECT * FROM pending_detections ORDER BY timestamp ASC LIMIT 100`.
3.  **Envio:** Monta um JSON com os 100 itens e envia para a API ViewSense/Supabase.
4.  **Commit:**
    *   **Sucesso (200 OK):** `DELETE FROM pending_detections WHERE id IN (...)`.
    *   **Falha (Erro de Rede/500):** Incrementa `retry_count` e espera um tempo (backoff exponencial) antes de tentar de novo.
    *   **Erro Fatal (400 Bad Request):** Move para uma tabela de `failed_detections` para análise manual (Dead Letter Queue).

### 4. Configuração (`config.yaml`)
Adicionar novos parâmetros para controlar esse comportamento:

```yaml
storage:
  db_path: "data/detections.db"
  max_retention_days: 7  # Limpar dados antigos se não conseguir enviar
  
sync:
  interval_seconds: 60   # Tentar enviar a cada 1 minuto
  max_batch_size: 200    # Ou quando acumular 200 detecções
```

## Benefícios Esperados
*   **Zero Data Loss:** Mesmo se a internet cair por horas, no momento que voltar, o worker esvazia a fila.
*   **Menor Custo de API:** Menos chamadas HTTP significam menos uso de computação no Edge Functions (Supabase cobra por invocação/cpu time).
*   **Desacoplamento:** O processamento de vídeo (pesado) fica totalmente separado da latência de rede.
