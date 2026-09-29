# PSLA4ML: SLAs Personalizados via _K-Skyband_ para Workflows de ML

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-5.2-FFD21E?logo=huggingface&logoColor=black)
![CodeCarbon](https://img.shields.io/badge/CodeCarbon-3.2.2-2E8B57?logo=codecarbon&logoColor=green)
![Tests](https://img.shields.io/badge/tests-651%20passing-brightgreen?logo=pytest&logoColor=white)

PSLA4ML mede execuções de aprendizado de máquina, explora configurações de recursos e hiperparâmetros e seleciona trade-offs por k-Skyband. Toda execução é um **workflow rastreável**, com tarefas, artefatos versionados, telemetria e manifesto persistido.

Referência: Santos, G.; Bedo, M.; Frota, Y.; Oliveira, D. *Definição de Acordos de Nível de Serviço Personalizados para Treinamento de Modelos via Consultas k-Skyband*. SBBD 2026. [DOI](https://doi.org/10.5753/sbbd.2026.249138).

## Instalação

Pré-requisitos: Python `>=3.12,<3.14` e [uv](https://docs.astral.sh/uv/). GPU NVIDIA e TPU são opcionais.

```bash
git clone https://github.com/gassantos/gridsearch-skyband.git
cd gridsearch-skyband
uv sync --group dev

# Opcional: suporte TPU
uv sync --extra tpu --group dev

# Verificação
uv run python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
uv run --group dev pytest
```

A suíte atual possui `651 passed, 4 skipped`.

## Notação de Workflow

Uma execução é definida por `ExperimentDefinition`, composta por `TaskDefinition`s. Tarefas declaram atividade, requisitos de recurso, assinaturas de entrada e artefatos de entrada/saída. Dependências são explícitas (`depends_on`) ou inferidas pelo fluxo de artefatos versionados.

| Notação | Atividade | Regime típico | Papel |
| --- | --- | --- | --- |
| T0 | `ingestion` | `build` | Ingestão, preparação, tokenização ou conversão de dados. |
| T1 | `pretraining` | `build` | Pré-treinamento que atualiza pesos. |
| T2 | `adaptation` | `build` | Fine-tuning ou treinamento que atualiza pesos. |
| T3 | `indexing` | `build` | Construção ou atualização de índice. |
| T4 | `retrieval_generation` | `service` | Recuperação e geração orientadas a latência. |
| T5 | `evaluation_monitoring` | `build` | Avaliação, qualidade e monitoramento. |

| Tipo de artefato | Enum | Exemplos |
| --- | --- | --- |
| Dados | `data` | dataset, corpus, features, métricas |
| Modelo | `model` | checkpoint, adaptador, pesos |
| Índice | `index` | índice vetorial ou invertido |
| Interação | `interaction` | predições, respostas, recuperação |

O fluxo local e o fluxo Hugging Face usam o ciclo mínimo T0 -> T2 -> T5. O manifesto de cada execução é gravado em `output/experiments/metrics/workflow_runs/<run-id>/`.

### Modos de execução

`single` e `grid` são **modos de orquestração**, não tipos de workflow.

| Modo | Comportamento |
| --- | --- |
| `--mode single` | Executa uma instância de workflow com `parallel_workers=1`. |
| `--mode grid` | Executa uma instância de workflow por combinação da grade; `--parallel` controla o paralelismo entre combinações. |
| `--workflow bertpli` | Executa o DAG especializado BERT-PLI. |
| `--workflow generic` | Executa uma especificação declarativa para `ml_classic`, `deep_learning`, `nlp` ou `llm`. |

No grid, cada combinação persiste seu próprio `ExperimentRun` antes de projetar `resources` e `evaluation` no schema consumido por Skyband e PSLA4ML.

## CLI

O ponto de entrada é:

```bash
uv run python -m main --help
```

### Single local

Usa o dataset e o loader declarados no `.config`, mas sempre executa o workflow T0 -> T2 -> T5:

```bash
uv run python -m main --mode single --no-skyband

uv run python -m main --mode single \
  --config config/experiments/BertPLI.config \
  --train-dataset train_task2_v3 \
  --gpu 0 --precision fp16 --no-skyband
```

### Single com Hugging Face

```bash
# Dataset do Hugging Face Hub
uv run python -m main --mode single \
  --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc \
  --no-skyband

# JSONL local via HuggingFace Datasets
uv run python -m main --mode single \
  --dataset-source local_json --train-dataset train_task2_v3 \
  --no-skyband
```

`--dataset-source hub` exige `--dataset-id`. As opções `hub` e `local_json` configuram `HuggingFaceDataset` para treino, validação e teste.

### Cache e retomada de single

O cache usa assinaturas semânticas de dataset, modelo, hiperparâmetros, seed, versão de métricas e revisão Git. A retomada só reutiliza tarefas cuja definição continua compatível.

```bash
uv run python -m main --mode single \
  --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc \
  --workflow-cache-dir output/experiments/workflow_cache \
  --workflow-resume-run output/experiments/metrics/workflow_runs/<run-id> \
  --no-skyband
```

### Grid search

Cada combinação executa T0 -> T2 -> T5. O paralelismo é entre combinações, não entre tarefas dependentes da mesma combinação.

```bash
# Grade de teste
uv run python -m main --mode grid \
  --grid-config gridsearch/config/grid_search_test.json \
  --parallel 2

# Grade com dataset do Hub
uv run python -m main --mode grid \
  --grid-config gridsearch/config/grid_search_test.json \
  --dataset-source hub --dataset-id nyu-mll/glue --dataset-config mrpc \
  --parallel 2

# GPUs explícitas e retomada de grade
uv run python -m main --mode grid \
  --grid-config gridsearch/config/grid_search_test.json \
  --gpu 0 1 --parallel 2 --resume
```

O estado incremental é salvo em `output/experiments/grid_search/grid_search_state_<device>_<date>.json`.

### Workflows especializados

#### BERT-PLI

O workflow BERT-PLI declara fine-tuning BERT, extração e conversão de features, treinamento Attention-RNN, inferência e avaliação de recuperação.

```bash
# Executar BERT-PLI
uv run python -m main --workflow bertpli --gpu 0

# Validar DAG e comandos sem treino
uv run python -m main --workflow bertpli --workflow-dry-run --gpu 0
```

#### Workflow genérico

Use para `ml_classic`, `deep_learning`, `nlp` ou `llm`. A especificação deve declarar pelo menos `ingestion`, `adaptation` e `evaluation_monitoring`, nessa ordem. Tarefas adicionais podem refinar essas atividades, mas `custom` não é aceito.

```bash
# Validar exemplo sem comandos externos
uv run python -m main --workflow generic \
  --workflow-spec examples/workflow/nlp.json \
  --workflow-dry-run

# Executar uma especificação própria
uv run python -m main --workflow generic --workflow-spec workflow.json
```

Exemplos: `examples/workflow/ml_classic.json`, `deep_learning.json`, `nlp.json` e `llm.json`.

Exemplo mínimo:

```json
{
  "name": "text-classification",
  "experiment_type": "nlp",
  "monitoring": {"enable_emissions": true, "environment_cost_per_hour_usd": 1.2},
  "tasks": [
    {
      "task_id": "ingest",
      "name": "Ingerir dados",
      "command": ["python", "prepare.py"],
      "activity": "ingestion"
    },
    {
      "task_id": "adapt",
      "name": "Adaptar modelo",
      "command": ["python", "train.py"],
      "depends_on": ["ingest"],
      "activity": "adaptation",
      "resources": {"gpu_count": 1, "coupling_degree": 0.9}
    },
    {
      "task_id": "evaluate",
      "name": "Avaliar modelo",
      "command": ["python", "evaluate.py"],
      "depends_on": ["adapt"],
      "activity": "evaluation_monitoring",
      "metrics_file": "output/evaluation.json"
    }
  ]
}
```

`metrics_file` deve ser um objeto JSON compatível com as métricas da tarefa:

```json
{
  "resources": {"total_gflops": 123.4},
  "evaluation": {"precision": 0.91, "recall": 0.88, "f1_score": 0.89, "accuracy": 0.90}
}
```

## Recursos, Telemetria e TPU

Use `--gpu ID [ID ...]` para selecionar GPUs. Sem a opção, o runtime detecta o dispositivo. Para TPU PJRT, instale o extra `tpu`, exporte `PJRT_DEVICE=TPU` e use BF16 quando o config original usar FP16.

```bash
export PJRT_DEVICE=TPU
uv run python -m main --mode single \
  --config config/experiments/BertPLI.config \
  --precision bf16 --tpu-cores 8 --no-skyband

uv run python -m scripts.homologate_tpu --expected-cores 8
```

Para TPU multicore no grid, use `--parallel 1`; os cores TPU são usados dentro de cada combinação.

`TaskTelemetryCollector` registra por tentativa tempo, RAM/VRAM, energia, emissões e custo. As tarefas também preservam métricas de domínio, como `total_gflops`, checkpoint, proveniência de dataset e qualidade. Para habilitar emissões:

```ini
[monitoring]
enable_monitoring = true
```

O custo pode ser derivado de `ENERGY_COST_USD_PER_KWH` ou da taxa horária declarada no ambiente do grid.

O manifesto em `workflow_runs` é a fonte de verdade da telemetria. Cada resultado em
`grid_search_state_*.json` replica essa estrutura em `task_telemetry`, com atividade,
regime, status e tentativas de cada tarefa; o campo `resources` continua como resumo
compatível com Skyband, derivado dessas tentativas. O `experiment_summary_*.csv` usa
uma linha por tentativa de tarefa, com as métricas padronizadas em colunas e contexto,
artefatos e métricas adicionais em campos JSON.

## Skyband, SLA e Tiers

Skyband é executado automaticamente após `single` ou `grid`; use `--no-skyband` para desativá-lo.

```bash
# Somente análise de estado existente
uv run python -m main --skyband-only

# Perfil de SLA e comparação com ranking escalar
uv run python -m main --skyband-only --sla-profile sustentavel --skyband-compare

# Restrições explícitas
uv run python -m main --skyband-only \
  --skyband-k 2 \
  --sla-constraint cost_usd=5.0 \
  --sla-constraint train_time_sec=7200

# Estado e métricas explícitos
uv run python -m main --skyband-only \
  --skyband-state output/experiments/grid_search/grid_search_state_CPU_<date>.json \
  --skyband-metrics train_time_sec cost_usd energy_kwh
```

Perfis em `gridsearch/config/sla_profiles.json`: `economico`, `sustentavel`, `tempo`, `balanceado`, `dev` e `producao`.

`--sla-constraint` aceita `train_time_sec`, `energy_kwh`, `peak_ram_mb`, `emissions_kg_co2` e `cost_usd`. Métricas típicas de `--skyband-metrics` são `train_time_sec`, `energy_kwh`, `total_gflops`, `emissions_kg_co2` e `cost_usd`; métricas de qualidade como `f1_score` e `accuracy` podem participar quando presentes.

`gridsearch/tiers.py` discretiza traces observados em `Tier`s e gera `TrainingTemplate`s. `gridsearch/milp_instance.py` mapeia dados persistidos para `PSLA4MLData`, mas ainda não resolve o MILP CC-IP.

## Estrutura

```
📁 gridsearch-skyband/
│
├── 🐍 main.py          CLI principal
├── 📁 cli/             Parser, comandos e orquestração
├── 📁 experiment/      Workflow, executor, telemetria, cache e persistência
├── 📁 gridsearch/      Grade, recursos, SLA, Skyband, tiers e instância MILP
├── 📁 dataset/         DataLoaders locais e Hugging Face
├── 📁 model/           Modelos e componentes de treino
├── 📁 formatter/       Formatação de entradas
├── 📁 tools/           Entrypoints de treino, teste e transformação
├── 📁 scripts/         Entrypoints de treino, teste e transformação
├── 📁 examples/        Especificações genéricas canônicas (workflow/)
├── 📁 tests/           Suíte pytest
└── 📁 docs/            Documentação complementar
```

Documentação complementar: [Workflow genérico](docs/GENERIC_WORKFLOW.md), [Grid Search](docs/GRIDSEARCH.md) e [Guia de execução](docs/EXECUTION_GUIDE.md).
