"""Testes de detecção e normalização de ambiente de execução."""

from experiment.helpers import detect_execution_environment, normalize_environment_label


def test_normalize_environment_label_maps_supported_values():
    assert normalize_environment_label("local") == "LOCAL"
    assert normalize_environment_label("CoLaB") == "COLAB"
    assert normalize_environment_label("gcp") == "GCP"
    assert normalize_environment_label("azure") == "AZURE"
    assert normalize_environment_label("aws") == "AWS"
    assert normalize_environment_label("AWS") == "AWS"


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
