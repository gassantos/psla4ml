"""
CLI — Pacote de interface de linha de comando (Facade)
=======================================================

Re-exporta a API pública do pacote CLI para manter
compatibilidade retroativa com leitores do ``main.py``.

Submódulos:
    - ``constants``   — Valores default de configuração
    - ``parser``      — Construção do ``ArgumentParser``
    - ``commands``    — Command Pattern (ABC + concretos)
    - ``runners``     — Funções de execução (single, grid, skyband)
    - ``sla_summary`` — Resumo SLA de execução

Autor: Gustavo Alexandre
"""

from .commands import (  # noqa: F401
    Command,
    GridCommand,
    SingleCommand,
    SkybandOnlyCommand,
    _resolve_command,
)
from .constants import (  # noqa: F401
    DEFAULT_CONFIG,
    DEFAULT_GRID_CONFIG,
    DEFAULT_MODE,
    DEFAULT_PARALLEL,
    DEFAULT_SKYBAND_K,
    DEFAULT_SLA_PROFILES,
    DEFAULT_TRAIN_DATASET,
)
from .parser import build_argument_parser  # noqa: F401
from .runners import (  # noqa: F401
    _build_dataset_overrides,
    _load_sla_profile,
    _parse_sla_constraints,
    run_grid_search_experiments,
    run_single_experiment,
    run_skyband_analysis,
    validate_paths,
)
from .sla_summary import (  # noqa: F401
    _build_execution_kpi_lines,
    _build_sla_execution_summary_lines,
    _emit_sla_execution_summary,
    _load_latest_grid_state,
)
