# Guia de Execução e Validação Funcional

**Projeto:** GridSearch-Skyband  
**Data de referência:** 08/08/2026  
**Objetivo:** orientar a execução manual e a validação de todas as funcionalidades implementadas, antes da atualização do README.

> Este documento é um protocolo de execução. Os comandos abaixo não foram executados durante sua elaboração. Cada resultado deve ser registrado somente após a execução no ambiente indicado.

## 1. Estratégia de validação

A validação está dividida em três níveis:

1. **Teste automatizado:** verifica regras isoladas e regressões com `pytest`.
2. **Validação funcional:** executa o fluxo real com uma grade pequena e inspeciona os artefatos.
3. **Homologação de infraestrutura:** comprova o uso efetivo de GPU ou TPU por evidências do runtime.

A aprovação de um teste unitário não comprova, isoladamente, que o treinamento ocorreu em um acelerador real. Da mesma forma, não é necessário executar as 4.860 configurações da grade completa para validar o mecanismo de Grid Search. A grade mínima ou de teste valida o fluxo; a grade completa constitui uma campanha experimental.

## 2. Pré-requisitos gerais

### 2.1 Ambientes suportados

| Ambiente | Finalidade |
| --- | --- |
| Windows, Linux ou macOS com CPU | testes, análise de estados e execução funcional de pequeno porte |
| Linux com NVIDIA CUDA | treinamento e Grid Search acelerados por GPU |
| Google Colab com TPU | homologação PyTorch/XLA e PJRT multicore |

### 2.2 Software

- Python `>=3.12,<3.14`.
- `uv` instalado.
- Git instalado.
- Acesso à internet para instalar dependências e baixar modelos ou datasets do Hugging Face Hub.
- Espaço disponível em `output/` para métricas, checkpoints e estados do Grid Search.

### 2.3 Preparação do projeto

1. Clone o repositório e acesse sua raiz:

```bash
git clone https://github.com/gassantos/gridsearch-skyband.git
cd gridsearch-skyband
```

1. Instale as dependências de produção e desenvolvimento:

```bash
uv sync --group dev
```

1. Em uma TPU do Google Colab, instale também o extra TPU:

```bash
pip install uv
uv sync --extra tpu --group dev
```

1. Confirme que o CLI está acessível:

```bash
uv run python -m main --help
```

**Critério de aprovação:** o comando deve encerrar com código zero e listar, entre outras, as opções `--mode`, `--grid-config`, `--parallel`, `--tpu-cores`, `--precision`, `--skyband-only`, `--sla-profile` e `--dataset-source`.

## 3. Convenções de evidência

Após cada execução, registre:

- data e ambiente;
- commit avaliado (`git rev-parse HEAD`);
- comando utilizado;
- código de saída;
- caminho dos artefatos;
- resultado observado;
- situação: aprovado, reprovado ou bloqueado.

Modelo de registro:

```md
- Funcionalidade:
- Ambiente:
- Commit:
- Comando:
- Código de saída:
- Artefatos:
- Evidência observada:
- Situação:
```

## 4. Testes automatizados

### 4.1 Suíte completa

1. Garanta que as dependências de desenvolvimento foram instaladas.
2. Execute:

```bash
uv run --group dev pytest
```

1. Opcionalmente, gere cobertura:

```bash
uv run --group dev pytest --cov=. --cov-report=term-missing
```

**Critérios de aprovação:** código de saída zero, nenhuma falha e quantidade final de testes registrada no relatório de validação. O número de testes deve ser reportado a partir da execução corrente, sem reutilizar contagens históricas.

### 4.2 Testes focados por funcionalidade

```bash
# Dispositivo CPU, CUDA, MPS e ramo TPU simulado
uv run --group dev pytest tests/test_device.py -v

# Métricas de avaliação, incluindo F1 e acurácia
uv run --group dev pytest tests/test_eval_metrics.py -v

# Pareto, Skyband, SLA e comparação com ranking escalar
uv run --group dev pytest tests/test_skyband.py -v

# Tiers, TrainingTemplate e geração PSLA4ML
uv run --group dev pytest tests/test_psla4ml.py -v

# Configurações e execução do Grid Search
uv run --group dev pytest tests/test_gridsearch.py tests/test_gridsearch_config.py -v

# Checkpoints convencionais e XLA
uv run --group dev pytest tests/test_checkpoint.py tests/test_xla_checkpoint.py -v

# Diagnóstico e homologação TPU por dados controlados
uv run --group dev pytest tests/test_tpu_check.py tests/test_tpu_homologation.py -v

# Dataset Hugging Face
uv run --group dev pytest tests/test_huggingface_dataset.py -v

# Regressões do runner central
uv run --group dev pytest tests/test_run_experiment_fixes.py -v
```

**Critério de aprovação:** todos os testes selecionados devem passar. Testes com mocks de TPU validam a lógica, mas não homologam hardware real.

## 5. Experimento único pelo runner central

### 5.1 Execução em CPU ou detecção automática

1. Escolha uma configuração em `config/experiments/`.
2. Execute sem Skyband, pois o modo `single` não produz estado de Grid Search:

```bash
uv run python -m main --mode single \
    --config config/experiments/BertPLI.config \
    --no-skyband
```

No PowerShell, use uma linha única ou substitua `\` pelo acento grave de continuação.

1. Verifique `output/experiments/metrics/`.
1. Abra o JSON mais recente e confira as seções `experiment`, `environment`, `hyperparameters`, `resources`, `evaluation` e `logs`.

**Critérios de aprovação:** código de saída zero; `experiment.status` igual a `success`; métricas numéricas persistidas; identificação coerente do dispositivo; ausência de erro fatal em `logs.stderr_tail`.

### 5.2 Seleção explícita de GPU

```bash
# Uma GPU
uv run python -m main --mode single --gpu 0 --no-skyband

# Mais de uma GPU disponível ao launcher
uv run python -m main --mode single --gpu 0 1 --no-skyband
```

**Pré-condição:** PyTorch deve reconhecer CUDA e os IDs informados devem existir.

**Critério de aprovação:** o JSON deve identificar GPU/CUDA no bloco `environment`. A mera aceitação da flag não comprova uso efetivo; confirme também utilização e memória com a ferramenta do ambiente, como `nvidia-smi`, durante uma execução feita manualmente.

### 5.3 Precisão configurável

```bash
uv run python -m main --mode single --precision fp32 --no-skyband
uv run python -m main --mode single --precision fp16 --gpu 0 --no-skyband
uv run python -m main --mode single --precision bf16 --no-skyband
```

**Restrições:** `fp16` deve ser usado somente em backend compatível. Em TPU/XLA, use `bf16`.

**Critério de aprovação:** o bloco `environment.precision` do resultado deve corresponder ao override solicitado e o treinamento deve terminar com sucesso.

### 5.4 Seleção do dataset local predefinido

```bash
uv run python -m main --mode single \
    --train-dataset train_task2_v2 \
    --no-skyband
```

Valores permitidos: `train_task2`, `train_task2_v2` e `train_task2_v3`.

**Critério de aprovação:** o resultado deve registrar o dataset selecionado e concluir o treinamento com sucesso.

## 6. Dataset Hugging Face

### 6.1 Dataset público do Hub

1. Escolha um dataset compatível com o formatter e o modelo.
2. Informe o ID e, quando necessária, a subconfiguração:

```bash
uv run python -m main --mode single \
    --dataset-source hub \
    --dataset-id nyu-mll/glue \
    --dataset-config mrpc \
    --no-skyband
```

**Critérios de aprovação:** download/carregamento concluído; splits reconhecidos; treinamento finalizado; resultado registra a origem utilizada.

> Compatibilidade semântica do schema deve ser verificada. Conseguir carregar um dataset não garante que seus campos correspondam aos campos esperados pelo formatter do modelo.

### 6.2 JSONL local via Hugging Face Datasets

1. Prepare os arquivos JSONL exigidos pela configuração de dados.
2. Caso o caminho precise ser sobrescrito, informe-o em `--dataset-id`:

```bash
uv run python -m main --mode single \
    --dataset-source local_json \
    --dataset-id caminho/para/os/dados \
    --no-skyband
```

1. Confirme no log os overrides `train_dataset_type`, `valid_dataset_type` e `test_dataset_type` para `HuggingFace`.

**Critério de aprovação:** os splits locais devem ser carregados e o experimento deve terminar com `status=success`.

## 7. Grid Search

### 7.1 Validação mínima do pipeline

A configuração mínima contém três combinações, apesar de uma nota interna antiga mencionar uma combinação.

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_minimal.json \
    --parallel 1 \
    --no-skyband
```

**Artefatos esperados em `output/experiments/grid_search/`:**

- `grid_search_state_*.json`;
- resultados consolidados do Grid Search;
- resumo textual, quando gerado pelo fluxo;
- métricas individuais em `output/experiments/metrics/`.

**Critérios de aprovação:** três combinações planejadas; resultados persistidos; ao menos um resultado com `status=success`; estado apto a retomada.

### 7.2 Grade de teste

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_test.json \
    --parallel 2
```

A análise Skyband é executada automaticamente ao final, salvo uso de `--no-skyband`.

**Critérios de aprovação:** oito combinações processadas ou justificadamente filtradas; estado persistido; relatório Skyband exibido quando existirem resultados bem-sucedidos.

### 7.3 Grade de qualidade preditiva

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_quality.json \
    --parallel 2
```

Essa grade combina parâmetros de recursos e qualidade (`learning_rate`, `batch_size`, `optimizer`, `dropout`, `max_seq_length` e `num_epochs`), totalizando 144 configurações antes dos filtros de SLA.

**Critérios de aprovação:** 144 combinações planejadas antes de filtros; resultados contêm F1 e acurácia quando a avaliação as produz; diferenças entre comprimentos de sequência, épocas e parâmetros de recursos ficam rastreáveis em `grid_params` ou `hyperparameters`.

### 7.4 Grade completa

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search.json \
    --parallel 4
```

A cardinalidade correta é:

$$
5 \times 3 \times 4 \times 3 \times 3 \times 3 \times 3 = 4.860.
$$

**Atenção:** esta é uma campanha experimental de alto custo. Valide primeiro as grades mínima, teste e qualidade. Ajuste `--parallel` à RAM e às GPUs disponíveis.

**Critério de aprovação:** 4.860 combinações geradas antes do pré-filtro; toda combinação deve terminar, ser rejeitada pelo SLA com justificativa ou permanecer registrada para retomada.

### 7.5 Grade multiambiente

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_multienv.json \
    --parallel 2
```

A grade gera 9.720 combinações de hiperparâmetros e ambientes ativos.

**Critérios de aprovação:** cada resultado deve registrar `grid_params.environment` ou `selected_environment`; custos e estimativas devem usar os dados do ambiente correspondente; total planejado igual a 9.720 antes dos filtros.

> Os aliases de ambiente representam cenários de custo e baseline. Eles não provisionam automaticamente recursos em nuvem nem transferem a execução para outro provedor.

### 7.6 Distribuição de GPUs no Grid Search

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_test.json \
    --parallel 2 \
    --gpu 0 1
```

**Critério de aprovação:** workers distribuídos entre os IDs informados, sem colisões fatais de memória. Confirme a distribuição por logs, JSONs de ambiente e monitoramento da GPU.

### 7.7 Retomada de execução

1. Inicie uma grade pequena ou de teste.
2. Interrompa-a de forma controlada somente após existir um estado persistido.
3. Retome usando a mesma configuração:

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_test.json \
    --parallel 2 \
    --resume
```

**Critérios de aprovação:** resultados concluídos anteriormente não devem ser executados novamente; combinações pendentes devem continuar; o estado final deve preservar resultados anteriores e novos.

## 8. Pré-filtro por SLA

### 8.1 Perfil predefinido

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_test.json \
    --parallel 2 \
    --sla-profile dev
```

Perfis disponíveis: `economico`, `sustentavel`, `tempo`, `balanceado`, `dev` e `producao`.

### 8.2 Restrições manuais

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_test.json \
    --parallel 2 \
    --sla-constraint peak_ram_mb=8192 \
    --sla-constraint train_time_sec=3600
```

### 8.3 Inspeção da evidência

1. Abra `grid_search_state_*.json`.
2. Localize `sla_prefilter`.
3. Confira contagens de aceitação e rejeição.
4. Inspecione `rejected_samples`, `estimated_value`, `threshold`, `metric` e `params`.
5. Confira `non_evaluable_constraints`.

**Regras esperadas:**

- `peak_ram_mb` pode ser estimada antes da execução;
- `train_time_sec` pode ser estimada quando a grade fornece baseline;
- energia, emissões e custo podem permanecer não avaliáveis antes da execução e ser aplicados na análise posterior;
- restrição não estimável não deve rejeitar silenciosamente uma combinação.

**Critério de aprovação:** toda rejeição deve possuir métrica, estimativa, limite e parâmetros auditáveis; constraints não avaliáveis devem ser explicitamente registradas.

## 9. Skyband e qualidade preditiva

### 9.1 Pré-condição

Skyband requer um `grid_search_state_*.json` com resultados `status=success`. O modo `single` não cria esse estado.

### 9.2 Frente de Pareto

```bash
uv run python -m main --skyband-only
```

**Critério de aprovação:** relatório `SKYBAND (k=1)`; candidatos admissíveis; `domination_count=0` para a frente de Pareto.

### 9.3 Skyband de ordem k

```bash
uv run python -m main --skyband-only --skyband-k 2
```

**Critério de aprovação:** somente resultados dominados por menos de dois outros devem integrar o conjunto retornado.

### 9.4 Estado específico e métricas customizadas

```bash
uv run python -m main --skyband-only \
    --skyband-state output/experiments/grid_search/grid_search_state_SELECIONADO.json \
    --skyband-k 2 \
    --skyband-metrics train_time_sec cost_usd energy_kwh
```

Substitua `grid_search_state_SELECIONADO.json` por um arquivo existente.

**Critério de aprovação:** o log deve indicar exatamente o arquivo escolhido e o relatório deve listar apenas as métricas solicitadas.

### 9.5 SLA pós-execução

```bash
uv run python -m main --skyband-only \
    --skyband-k 2 \
    --sla-constraint cost_usd=5.0 \
    --sla-constraint train_time_sec=7200
```

**Critério de aprovação:** nenhum candidato do relatório pode exceder as constraints informadas.

### 9.6 Perfil de SLA e comparação com ranking escalar

```bash
uv run python -m main --skyband-only \
    --sla-profile balanceado \
    --skyband-compare
```

**Critérios de aprovação:** relatório Skyband seguido da seção `SKYBAND vs RANKING ESCALAR`, contendo similaridade de Jaccard, interseção e diferenças entre os conjuntos.

### 9.7 F1 e acurácia na dominância

A CLI pública limita `--skyband-metrics` às métricas de recursos descritas no parser. A inclusão conjunta de `f1_score` e `accuracy` está exposta pela API PSLA4ML por `include_quality_metrics=True`.

Valide-a pelo procedimento programático da Seção 10 e confirme que:

- métricas de recursos são minimizadas;
- F1 e acurácia são maximizadas;
- traces sem métricas de qualidade não são tratados como melhores por ausência de valor.

## 10. PSLA4ML, tiers e TrainingTemplate

Não existe uma flag CLI dedicada para gerar tiers. A execução correta usa a API `gridsearch.tiers` sobre um estado real do Grid Search.

### 10.1 Preparação

1. Execute primeiro uma grade pequena, de teste ou de qualidade.
2. Identifique o `grid_search_state_*.json` que será analisado.
3. Crie temporariamente um script, por exemplo `validate_psla4ml.py`, com o conteúdo abaixo:

```python
import json
from pathlib import Path

from gridsearch.tiers import TrainingTemplate, generate_psla4ml

state_file = Path(
    "output/experiments/grid_search/grid_search_state_SELECIONADO.json"
)
with state_file.open(encoding="utf-8") as stream:
    state = json.load(stream)

results = [
    result for result in state["results"]
    if result.get("status") == "success"
]

template = TrainingTemplate(
    architecture="BertPLI",
    dataset_id="train_task2",
    fixed_hyperparams={"dropout": 0.1},
)

tiers = generate_psla4ml(
    results=results,
    k=2,
    metrics=[
        "train_time_sec",
        "energy_kwh",
        "emissions_kg_co2",
        "cost_usd",
    ],
    strategy="median",
    template=template,
)

for tier in tiers:
    print(tier.to_dict())

print(f"tiers={len(tiers)}")
```

1. Substitua o nome do estado e ajuste `architecture`, `dataset_id` e `dropout` aos valores realmente registrados.
1. Execute manualmente:

```bash
uv run python validate_psla4ml.py
```

### 10.2 Critérios de aprovação

- somente traces compatíveis com o template participam da consulta;
- cada tier contém modelo, dataset, hardware, hiperparâmetros, métricas brutas e discretizadas;
- os intervalos usam limiares calculados pela mediana, salvo estratégia diferente;
- tiers estão ordenados por `domination_count` crescente;
- todos os tiers possuem `domination_count < k`;
- conjunto vazio deve ser explicado por template, SLA ou ausência de resultados válidos, sem fabricar tiers.

### 10.3 PSLA4ML com qualidade preditiva

No mesmo script, substitua a chamada por:

```python
tiers = generate_psla4ml(
    results=results,
    k=2,
    metrics=[
        "train_time_sec",
        "energy_kwh",
        "emissions_kg_co2",
        "cost_usd",
        "f1_score",
        "accuracy",
    ],
    strategy="median",
    template=template,
    include_quality_metrics=True,
)
```

**Critério de aprovação:** `f1_score` e `accuracy` devem aparecer em `raw_metrics` e `discretized` quando presentes nos traces, sendo tratadas como objetivos de maximização na dominância.

## 11. Correlação e multicolinearidade

Não existe uma flag CLI específica. Use a API sobre resultados reais.

1. Crie temporariamente `validate_correlations.py`:

```python
import json
from pathlib import Path

from gridsearch.analysis import analyze_correlations, detect_collinear_metrics

state_file = Path(
    "output/experiments/grid_search/grid_search_state_SELECIONADO.json"
)
with state_file.open(encoding="utf-8") as stream:
    state = json.load(stream)

results = [
    result for result in state["results"]
    if result.get("status") == "success"
]

hp_correlations = analyze_correlations(results)
report = detect_collinear_metrics(
    results,
    metrics=[
        "train_time_sec",
        "energy_kwh",
        "emissions_kg_co2",
        "cost_usd",
    ],
    threshold=0.95,
)

print("Correlações hiperparâmetro x métrica:")
for name, value in sorted(hp_correlations.items()):
    print(name, value)

print("Matriz entre métricas:")
for metric, row in report.correlation_matrix.items():
    print(metric, row)

print("Pares colineares:")
for metric_a, metric_b, coefficient in report.collinear_pairs:
    print(metric_a, metric_b, coefficient)
```

1. Substitua o estado e execute:

```bash
uv run python validate_correlations.py
```

**Critérios de aprovação:** matriz simétrica com diagonal 1; coeficientes entre -1 e 1 ou `None` quando insuficientes; pares com $|r| \ge 0{,}95$ em `collinear_pairs`; número de amostras igual ao de resultados bem-sucedidos usados.

> Correlação exige variação e pelo menos duas observações válidas. Uma grade pequena pode ser insuficiente para conclusões científicas, mesmo que a função opere corretamente.

## 12. Monitoramento, energia, emissões e custo

### 12.1 CodeCarbon

1. Abra a configuração experimental utilizada.
2. Confirme:

```ini
[monitoring]
enable_monitoring = true
```

1. Execute um experimento `single` ou uma grade pequena.
1. Inspecione:

- `resources.energy_kwh`;
- `resources.emissions_kg_co2`;
- `output/experiments/metrics/EmissionsCO2_<device>_<YYYYMMDD>.csv`.

**Critério de aprovação:** valores e CSV são persistidos quando o monitoramento está ativo; indisponibilidade do CodeCarbon deve ser explicitamente registrada, não convertida em sucesso silencioso.

### 12.2 Tarifa energética

Linux/macOS:

```bash
ENERGY_COST_USD_PER_KWH=0.08 uv run python -m main --mode single --no-skyband
```

PowerShell:

```powershell
$env:ENERGY_COST_USD_PER_KWH = "0.08"
uv run python -m main --mode single --no-skyband
```

**Critério de aprovação:** o resultado deve refletir a tarifa configurada no campo de custo energético aplicável. Remova a variável ou restaure seu valor após a validação.

## 13. Checkpoints

### 13.1 Validação automatizada

```bash
uv run --group dev pytest tests/test_checkpoint.py tests/test_xla_checkpoint.py -v
```

### 13.2 Validação funcional

1. Execute um treinamento que produza checkpoint.
2. Confirme a criação de arquivo em `output/checkpoints/`.
3. Verifique se o checkpoint contém estado do modelo, otimizador, scheduler, época treinada e passo global, conforme aplicável.
4. Execute novamente o fluxo configurado para carregar esse checkpoint.
5. Confirme no log que a restauração ocorreu e que o treinamento continuou do estado esperado.

**Critérios de aprovação:** checkpoint legível; restauração sem incompatibilidade; `trained_epoch` e `global_step` preservados; em XLA, salvamento realizado pelo processo apropriado e artefato reutilizável.

## 14. TPU, BF16, PJRT e observabilidade XLA

Esta seção deve ser executada em Google Colab com runtime TPU selecionado.

### 14.1 Preparação do Colab

1. Selecione **Runtime > Change runtime type > TPU**.
2. Clone o repositório e acesse sua raiz.
3. Instale as dependências:

```bash
pip install uv
uv sync --extra tpu --group dev
```

1. Defina PJRT:

```bash
export PJRT_DEVICE=TPU
```

1. Opcionalmente, confirme a importação:

```bash
uv run python -c "import torch_xla; print(torch_xla.__version__)"
```

### 14.2 Modo single em um core lógico solicitado

```bash
uv run python -m main --mode single \
    --config config/experiments/BertPLI.config \
    --precision bf16 \
    --tpu-cores 1 \
    --no-skyband
```

Use a expressão “modo single com `--tpu-cores 1`”. Não conclua que a topologia física é single-core apenas pela flag.

### 14.3 TPU multicore

```bash
uv run python -m main --mode single \
    --config config/experiments/BertPLI.config \
    --precision bf16 \
    --tpu-cores 8 \
    --no-skyband
```

Para Grid Search em TPU multicore:

```bash
uv run python -m main --mode grid \
    --grid-config gridsearch/config/grid_search_minimal.json \
    --precision bf16 \
    --tpu-cores 8 \
    --parallel 1 \
    --no-skyband
```

**Regra:** `--tpu-cores` maior que 1 exige `--parallel 1`, evitando spawn aninhado. O paralelismo ocorre dentro do experimento, entre workers PJRT.

### 14.4 Homologação automática do resultado real

Após o treinamento multicore:

```bash
uv run python -m scripts.homologate_tpu --expected-cores 8
```

Para validar um JSON específico:

```bash
uv run python -m scripts.homologate_tpu \
    output/experiments/metrics/RESULTADO_TPU.json \
    --expected-cores 8
```

**Critérios obrigatórios:**

- `experiment.status == "success"`;
- `environment.device_type == "TPU"`;
- `execution.xla_world_size == 8`;
- `tpu_acceleration_check.accelerator_active == true`;
- `compile_count > 0`;
- `execute_count > 0`.

A mensagem `BL-08 homologado` e código de saída zero indicam que o JSON satisfaz esses critérios. Uma falha lista cada evidência ausente.

### 14.5 BF16 e formas estáticas

1. Use `--precision bf16` em TPU.
2. Inspecione o resultado e os logs para confirmar ausência de erro de FP16 em XLA.
3. Observe todos os batches, incluindo o último, em busca de recompilações excessivas ou erro de forma.
4. Execute também os testes de regressão:

```bash
uv run --group dev pytest \
    tests/test_run_experiment_fixes.py \
    tests/test_xla_checkpoint.py \
    tests/test_tpu_check.py \
    -v
```

**Critério de aprovação:** treinamento completo em BF16; sem falha de shape no último batch; métricas XLA positivas; checkpoint válido quando habilitado.

## 15. Isolamento de backend nos testes de dispositivo

O objetivo é assegurar que uma instalação de `torch_xla` não faça testes CPU/GPU selecionarem TPU indevidamente.

```bash
uv run --group dev pytest tests/test_device.py -v
```

**Critérios de aprovação:** ramos CPU, CUDA e MPS permanecem controláveis quando `_XLA_AVAILABLE` é falso; ramo TPU é selecionado apenas quando XLA está disponível e há dispositivo suportado; nenhum teste local exige hardware TPU real.

## 16. Ordem recomendada de validação integral

1. Preparação do ambiente e `--help`.
2. Suíte automatizada completa.
3. Experimento `single` em CPU ou GPU.
4. Dataset local predefinido.
5. Dataset Hugging Face Hub e JSONL local.
6. Grade mínima sem Skyband.
7. Retomada de uma grade interrompida.
8. Grade de teste com Skyband automático.
9. Pré-filtro SLA e auditoria do estado.
10. Skyband-only, SLA pós-execução e comparação escalar.
11. Grade de qualidade e métricas F1/acurácia.
12. PSLA4ML e TrainingTemplate via API.
13. Correlação e multicolinearidade via API.
14. CodeCarbon, tarifa e artefatos.
15. Checkpoints e restauração.
16. GPU explícita, quando disponível.
17. TPU single, TPU multicore e homologação XLA.
18. Somente após aprovação dos passos aplicáveis: executar campanhas completa e multiambiente, caso haja orçamento computacional.
19. Consolidar evidências e atualizar README e Status Report.

## 17. Matriz de aprovação

| Funcionalidade | Teste automatizado | Validação funcional | Homologação externa | Situação |
| --- | --- | --- | --- | --- |
| Runner `single` | `test_run_experiment_fixes.py` | JSON de experimento bem-sucedido | não aplicável | feito |
| Dataset local/HF | `test_huggingface_dataset.py` | treino com origem selecionada | acesso ao Hub, se usado | feito |
| Grid Search | `test_gridsearch*.py` | grade mínima/teste | não aplicável | feito |
| Retomada | testes de Grid/checkpoint | estado retomado sem repetição | não aplicável | pendente |
| Pré-filtro SLA | `test_skyband.py` e testes de Grid | `sla_prefilter` auditável | não aplicável | feito |
| Skyband | `test_skyband.py` | relatório sobre estado real | não aplicável | feito |
| F1/acurácia | `test_eval_metrics.py` e `test_psla4ml.py` | tiers com qualidade | não aplicável | feito |
| PSLA4ML | `test_psla4ml.py` | tiers sobre estado real | não aplicável | feito |
| Correlações | testes de análise aplicáveis | matriz sobre estado real | amostra suficiente | feito |
| CodeCarbon | testes aplicáveis | JSON e CSV de emissões | sensor/estimativa do ambiente | feito |
| Checkpoint | `test_checkpoint.py` | restauração real | não aplicável | feito |
| XLA/BF16 | testes XLA | treino TPU bem-sucedido | Colab TPU | feito |
| PJRT multicore | testes de launcher/homologação | world size esperado | Colab TPU com 8 workers | pendente |
| Observabilidade XLA | `test_tpu_check.py` | contagens XLA positivas | runtime TPU real | pendente |
