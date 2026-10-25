"""Testes de detecção e normalização de ambiente de execução."""

from experiment.helpers import detect_execution_environment, normalize_environment_label


def test_normalize_environment_label_maps_supported_values():
    assert normalize_environment_label("local") == "LOCAL"
    assert normalize_environment_label("CoLaB") == "COLAB"
    assert normalize_environment_label("gcp") == "GCP"
    assert normalize_environment_label("azure") == "AZURE"
    assert normalize_environment_label("aws") == "AWS"
    assert normalize_environment_label("") == "LOCAL"


def test_detect_execution_environment_defaults_to_local(monkeypatch):
    for key in (
        "COLAB_RELEASE_TAG",
        "COLAB_GPU",
        "COLAB_BACKEND_VERSION",
        "WEBSITE_SITE_NAME",
        "FUNCTIONS_WORKER_RUNTIME",
        "AZURE_HTTP_USER_AGENT",
        "IDENTITY_ENDPOINT",
        "GOOGLE_CLOUD_PROJECT",
        "GCP_PROJECT",
        "K_SERVICE",
        "FUNCTION_TARGET",
        "GAE_ENV",
        "AWS_EXECUTION_ENV",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "LAMBDA_TASK_ROOT",
        "ECS_CONTAINER_METADATA_URI",
        "ECS_CONTAINER_METADATA_URI_V4",
    ):
        monkeypatch.delenv(key, raising=False)

    assert detect_execution_environment() == "LOCAL"


def test_detect_execution_environment_prioritizes_colab(monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-test")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")

    assert detect_execution_environment() == "COLAB"


def test_execution_context_reports_detected_environment_and_machine(monkeypatch):
    for key in ("COLAB_RELEASE_TAG", "COLAB_GPU", "COLAB_BACKEND_VERSION"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-x")
    monkeypatch.setenv("PSLA4ML_MACHINE_NAME", "host-42")

    from experiment.helpers import execution_context

    context = execution_context()

    assert context.items() >= {
        "selected_environment": "host-42",
        "configured_environment": "COLAB",
        "resource_machine_name": "host-42",
    }.items()
    assert {"cpu_count", "gpu_count", "gpu_model", "disk_total_gb", "disk_free_gb"} <= set(context)


def test_hardware_info_reports_cpu_gpu_and_output_disk(monkeypatch):
    from experiment import helpers

    helpers._static_hardware.cache_clear()
    monkeypatch.setattr(helpers.psutil, "cpu_count", lambda logical=True: 12)
    monkeypatch.setattr(helpers.shutil, "disk_usage", lambda _path: type("U", (), {
        "total": 500 * 1024 ** 3, "free": 125.5 * 1024 ** 3,
    })())
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda _i: "NVIDIA T4")

    try:
        info = helpers.hardware_info()
    finally:
        helpers._static_hardware.cache_clear()

    assert info == {
        "cpu_count": 12, "gpu_count": 2, "gpu_model": "NVIDIA T4",
        "disk_total_gb": 500.0, "disk_free_gb": 125.5,
    }


def test_hardware_info_without_cuda_has_no_gpu(monkeypatch):
    from experiment import helpers

    helpers._static_hardware.cache_clear()
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    try:
        info = helpers.hardware_info()
    finally:
        helpers._static_hardware.cache_clear()

    assert info["gpu_count"] == 0 and info["gpu_model"] is None
    assert info["cpu_count"] >= 1 and info["disk_total_gb"] > 0
