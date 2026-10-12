import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def load_workflow(name):
    # BaseLoader preserves GitHub's "on" key instead of treating it as a YAML 1.1 boolean.
    return yaml.load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_bundle_identity_and_resource_includes():
    bundle = load_yaml("databricks.yml")
    assert bundle["bundle"]["name"] == "nyc_taxi_mlflow"
    assert bundle["bundle"]["uuid"]
    for pattern in bundle["include"]:
        paths = list(ROOT.glob(pattern))
        assert paths, f"Resource include matches no files: {pattern}"
        for path in paths:
            assert load_yaml(path.relative_to(ROOT))["resources"]
    assert bundle["artifacts"]["python_artifact"]["build"] == "uv build --wheel"


def test_targets_are_isolated():
    targets = load_yaml("databricks.yml")["targets"]
    assert targets["dev"]["default"] is True
    assert targets["dev"]["mode"] == "development"
    assert targets["dev"]["variables"]["schema"] == "${workspace.current_user.short_name}"
    for target, schema, expected_root in (
        ("ci_dev", "nyc_taxi_mlflow_dev", "~/.bundle/${bundle.name}/${bundle.target}"),
        ("prod", "nyc_taxi_mlflow_prod", "/Workspace/Shared/.bundle/${bundle.name}/${bundle.target}"),
    ):
        config = targets[target]
        assert config["variables"] == {"catalog": "dbx_mlflow_dev", "schema": schema}
        assert config["workspace"]["root_path"] == expected_root
        assert config["run_as"]["service_principal_name"] == "${var.service_principal_id}"
        assert config["permissions"] == [
            {"service_principal_name": "${var.service_principal_id}", "level": "CAN_MANAGE"}
        ]
    assert targets["prod"]["mode"] == "production"
    assert targets["prod"]["presets"]["trigger_pause_status"] == "PAUSED"
    assert {target["workspace"]["host"] for target in targets.values()} == {
        "https://adb-7405619604318859.19.azuredatabricks.net"
    }


def test_sample_workloads_are_paused_and_serverless():
    job = load_yaml("resources/sample_job.job.yml")["resources"]["jobs"]["sample_job"]
    assert job["trigger"]["pause_status"] == "PAUSED"
    assert all("job_cluster_key" not in task and "existing_cluster_id" not in task for task in job["tasks"])
    assert (ROOT / "src/sample_notebook.ipynb").is_file()
    assert job["environments"][0]["spec"]["dependencies"] == ["../dist/*.whl"]
    pipeline = load_yaml("resources/nyc_taxi_mlflow_etl.pipeline.yml")["resources"]["pipelines"]["nyc_taxi_mlflow_etl"]
    assert pipeline["serverless"] is True
    assert pipeline["catalog"] == "${var.catalog}"
    assert pipeline["schema"] == "${var.schema}"
    assert list((ROOT / "resources").glob("*.yml"))
    assert (ROOT / "resources" / pipeline["root_path"]).is_dir()


def test_offline_tests_do_not_initialize_databricks():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/unit"]
    assert not (ROOT / "tests/conftest.py").exists()
    assert (ROOT / "tests/integration/conftest.py").is_file()
    assert "databricks-connect" not in project["dependency-groups"]["ci"]


def test_pr_checks_have_no_fork_authentication():
    workflow = load_workflow("ci.yml")
    assert set(workflow["on"]) == {"pull_request"}
    assert workflow["on"]["pull_request"]["branches"] == ["main"]
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["validate"]
    assert job["if"] == "github.event.pull_request.head.repo.full_name == github.repository"
    assert job["needs"] == "checks"
    assert job["environment"] == "validation"
    assert job["permissions"]["id-token"] == "write"
    assert job["steps"][-1]["run"] == "databricks bundle validate --target ci_dev"
    offline = load_workflow("checks.yml")
    assert offline["permissions"] == {"contents": "read"}
    commands = [step["run"] for step in offline["jobs"]["offline"]["steps"] if "run" in step]
    assert "uv run --frozen --only-group ci pytest" in commands
    assert "uv build --wheel" in commands


def test_deployments_require_main_and_are_serialized():
    workflow = load_workflow("deploy.yml")
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in workflow["on"]
    assert '"$GITHUB_REF" = "refs/heads/main"' in workflow["jobs"]["main-only"]["steps"][0]["run"]
    assert workflow["jobs"]["checks"]["needs"] == "main-only"
    job = workflow["jobs"]["deploy"]
    assert job["needs"] == "checks"
    assert job["environment"] == "${{ github.event_name == 'workflow_dispatch' && 'prod' || 'dev' }}"
    assert job["env"]["BUNDLE_TARGET"] == "${{ github.event_name == 'workflow_dispatch' && 'prod' || 'ci_dev' }}"
    assert job["concurrency"]["cancel-in-progress"] == "false"
    commands = [step["run"] for step in job["steps"] if "run" in step]
    assert commands[-2:] == [
        'databricks bundle validate --target "$BUNDLE_TARGET"',
        'databricks bundle deploy --target "$BUNDLE_TARGET"',
    ]
    assert not any("bundle run" in command for command in commands)


def test_integration_is_manual_and_uses_explicit_serverless():
    workflow = load_workflow("integration.yml")
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert '"$GITHUB_REF" = "refs/heads/main"' in workflow["jobs"]["main-only"]["steps"][0]["run"]
    job = workflow["jobs"]["integration"]
    assert job["needs"] == "main-only"
    assert job["environment"] == "dev"
    assert job["env"]["DATABRICKS_SERVERLESS_COMPUTE_ID"] == "auto"
    assert job["steps"][-1]["run"] == "uv run --frozen pytest tests/integration -m integration"


@pytest.mark.parametrize("path", sorted((ROOT / ".github").rglob("*.yml")), ids=lambda path: path.name)
def test_actions_are_pinned_and_checkouts_do_not_persist_tokens(path):
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    jobs = data.get("jobs", {"composite": data.get("runs", {})})
    for job in jobs.values():
        for step in job.get("steps", []):
            action = step.get("uses", "")
            if action and not action.startswith("./"):
                assert re.fullmatch(r"[\w-]+/[\w-]+@[0-9a-f]{40}", action)
            if action.startswith("actions/checkout@"):
                assert step["with"]["persist-credentials"] == "false"
