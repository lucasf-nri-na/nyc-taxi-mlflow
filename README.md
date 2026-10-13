# nyc-taxi-mlflow

Databricks Declarative Automation Bundle generated with the official
`default-python` template using Databricks CLI 1.20.0. The bundle and Python
package are named `nyc_taxi_mlflow`.

This is project scaffolding, not a completed MLflow training or serving system.
It loads raw taxi trip data (`yellow_tripdata_2026-01.parquet`) into a raw table (`raw_taxis`),
downloading the parquet file only if the table does not exist or has zero records.
The sample notebook job, Python wheel task, and Lakeflow ETL pipeline operate on
this raw taxi table, and a dedicated model training job trains and logs
`RandomForestRegressor` and `XGBoostRegressor` models using MLflow.
All sample triggers are **paused**, including in
production. Deploying creates/updates resources; it does not run them.

## Layout

| Path | Purpose |
| --- | --- |
| `databricks.yml` | Bundle targets, artifact build, catalog/schema variables |
| `resources/` | Sample job, model training job, and serverless pipeline definitions |
| `src/nyc_taxi_mlflow/` | Shared Python package (taxis ingestion, training, and wheel entry points) |
| `src/sample_notebook.ipynb` | Sample notebook task |
| `src/nyc_taxi_mlflow_etl/` | Pipeline transformations |
| `tests/unit/` | Credential-free configuration and workflow checks |
| `tests/integration/` | Original sample Spark test and Databricks Connect fixtures |
| `fixtures/` | Test data fixtures |
| `.github/workflows/` | PR checks, deployment, and manual integration tests |

## Local development

Prerequisites: Git, uv 0.11.1, Python 3.12 (uv can install it), and the modern
Databricks CLI 1.20.0. Do not install the legacy `databricks-cli` pip package.
The generated dependency range uses Databricks Connect 16.4; `uv.lock` pins the
resolved dependencies. Local and CI Python versions must remain compatible with
Connect and the workspace's serverless environment.

From PowerShell in the repository root:

```powershell
uv sync --frozen --dev
uv run --frozen --only-group ci ruff check .
uv run --frozen --only-group ci pytest
uv build --wheel
databricks auth login --host https://adb-7405619604318859.19.azuredatabricks.net --profile nyc-taxi
databricks bundle validate --target dev --profile nyc-taxi
```

`--only-group ci` installs only offline check tools and can remove development
dependencies from `.venv`. Run `uv sync --frozen --dev` again before IDE/Connect
work. Default pytest discovery and VS Code test discovery use `tests/unit`;
they do not initialize Spark. Ruff suppresses undefined-name checks only for
the notebook and pipeline files where Databricks supplies runtime globals.

After merging and provisioning the required data permissions, deploy a personal
copy explicitly:

```powershell
databricks bundle deploy --target dev --profile nyc-taxi
```

Use a workspace/IDE workflow instead if preferred:
[workspace bundle authoring](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/workspace)
or the [Databricks VS Code extension](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/vscode-ext).

## Targets and isolation

All targets use workspace
`https://adb-7405619604318859.19.azuredatabricks.net` and catalog
`dbx_mlflow_dev`.

| Target | Mode | Schema | Deployment identity/location |
| --- | --- | --- | --- |
| `dev` (default) | development | Current user's short name | Personal authenticated identity and default user bundle path |
| `ci_dev` | development | `nyc_taxi_mlflow_dev` | Dev service principal; `~/.bundle/nyc_taxi_mlflow/ci_dev` |
| `prod` | production | `nyc_taxi_mlflow_prod` | Prod service principal; `/Workspace/Shared/.bundle/nyc_taxi_mlflow/prod` |

Shared targets require `service_principal_id` (the application/client ID, not
the numeric account principal ID). Workflows set
`BUNDLE_VAR_service_principal_id` from `DATABRICKS_CLIENT_ID` and verify that the
authenticated identity matches. This keeps the run identity independent of the
scaffold author's account. Missing IDs fail validation/deployment rather than
falling back to a personal identity.

Use separate dev and prod service principals. The CI resource prefix is
`[ci_dev]` and production prefix is `[prod]`; paths and schemas are separated.
They still share a catalog and workspace, so this is not infrastructure-level
production isolation. Production dispatch selects a revision of `main`; it does
not guarantee that revision is the most recently dev-deployed revision.

## GitHub Actions behavior

| Workflow | Trigger | Behavior |
| --- | --- | --- |
| Pull request CI | PR targeting `main` | Reusable offline checks for every PR; same-repository PRs also request approval for authenticated `ci_dev` validation |
| Offline checks | Reusable workflow | Frozen CI dependencies, Ruff, static config tests, wheel build and package-content check |
| Deploy bundle | Push to `main` | Checks, authenticated validation, then deployment of `ci_dev` through environment `dev` |
| Deploy bundle | Manual dispatch on `main` | Checks, authenticated validation, then deployment of `prod` after environment approval |
| Manual integration tests | Manual dispatch on `main` | Original taxi test using dev OIDC and serverless Databricks Connect |

Fork PRs never enter the authenticated validation job. No
`pull_request_target` workflow is used. Authentication uses GitHub OIDC with
job-scoped `id-token: write`; other jobs receive only `contents: read`.
Actions are pinned to commit SHAs, CLI/uv versions are explicit, and checkout
does not persist the GitHub token.

Deployment jobs are serialized per target and do not cancel an in-progress
deployment. Manual deployment and integration workflows reject non-`main`
refs. Neither workflow automatically runs sample jobs/pipelines or destroys
resources. Do not merge the bootstrap PR until the dev identity and permissions
are ready: merging triggers dev deployment.

Static YAML tests check project invariants; they are **not** a substitute for
authenticated `databricks bundle validate`.

## Platform setup

### GitHub settings

The bootstrap configures these GitHub environments and the workspace host
variable. Workflow YAML alone cannot configure environment protection rules.

| Environment | Allowed refs / gate | Variables |
| --- | --- | --- |
| `validation` | Same-repository PR job; required approval | `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID` for the dev/validation principal |
| `dev` | `main` branch only | `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID` for the dev principal |
| `prod` | `main` branch only; required approval | `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID` for the prod principal |

Approval reviewer is the repository owner; self-review is allowed for this
single-owner repository. Administrator bypass is disabled for approval-gated
environments. Revisit reviewer separation before expanding production use.
An environment-bound federation policy is not itself a PR trust restriction:
review the **entire PR**, including workflow/action changes, before approving
`validation`. Approval permits execution of that revision with the dev
principal's capabilities. Do not approve untrusted code.

Add the client IDs after identity provisioning:

```powershell
gh variable set DATABRICKS_CLIENT_ID --repo lucasf-nri-na/nyc-taxi-mlflow --env validation --body "<dev-application-id>"
gh variable set DATABRICKS_CLIENT_ID --repo lucasf-nri-na/nyc-taxi-mlflow --env dev --body "<dev-application-id>"
gh variable set DATABRICKS_CLIENT_ID --repo lucasf-nri-na/nyc-taxi-mlflow --env prod --body "<prod-application-id>"
```

These IDs and the host are configuration variables, not credentials.
Require the offline PR status check on `main` and a pull request before merging;
authenticated validation is deliberately not a required status for fork PRs.
Environment approvals and federation must be configured before authenticated
jobs are approved. Missing settings fail explicitly.

### Databricks service principals and federation

An account/workspace administrator must:

1. Create/assign dev and prod service principals to this workspace with permission
   to create/manage their bundle jobs and pipelines, use serverless compute, and
   manage their separate shared workspace deployment folders. Restrict their
   data privileges to the intended schemas. Validation uses the dev principal.
2. Record each principal's application ID, numeric account principal ID, and the
   Databricks account ID. Numeric IDs are for the account federation API; the
   application ID is for `DATABRICKS_CLIENT_ID` and bundle `run_as`.
3. Create service-principal federation policies using issuer
   `https://token.actions.githubusercontent.com`, the Databricks account ID as
   audience, and the exact environment-specific OIDC subjects below.
4. Configure the GitHub environment client ID variables and complete authenticated
   validation before merging. Production can remain disabled until its setup is
   complete; do not dispatch it prematurely.

This repository uses GitHub's **immutable subject** format, verified through
`GET /repos/lucasf-nri-na/nyc-taxi-mlflow/actions/oidc/customization/sub`:

```text
repo:lucasf-nri-na@229027340/nyc-taxi-mlflow@1415926339:environment:validation
repo:lucasf-nri-na@229027340/nyc-taxi-mlflow@1415926339:environment:dev
repo:lucasf-nri-na@229027340/nyc-taxi-mlflow@1415926339:environment:prod
```

Use the first two subjects on the dev principal and the last on the prod
principal. Do not substitute the older name-only format. Recheck GitHub's OIDC
configuration if this repository is renamed, transferred, or customized.

Example policy file (replace account ID; set the subject for each environment):

```json
{
  "oidc_policy": {
    "issuer": "https://token.actions.githubusercontent.com",
    "audiences": ["<databricks-account-id>"],
    "subject": "repo:lucasf-nri-na@229027340/nyc-taxi-mlflow@1415926339:environment:dev"
  }
}
```

Save the populated JSON outside the repository as `federation-policy.json` and
apply using a separately authenticated **account** profile:

```powershell
databricks account service-principal-federation-policy create <numeric-principal-id> --profile account-admin --json '@federation-policy.json'
```

No personal access token or client secret is needed in GitHub. See the official
[GitHub OIDC setup](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/auth/provider-github).

### Data and compute prerequisites

`dbx_mlflow_dev` exists, but the bootstrap does not create schemas or grant
Databricks privileges. An administrator must create
`dbx_mlflow_dev.nyc_taxi_mlflow_dev`,
`dbx_mlflow_dev.nyc_taxi_mlflow_prod`, and each developer's personal schema.
Grant `USE CATALOG`, `USE SCHEMA`, and the table/materialized-view creation and
management privileges required by the sample pipeline and ingestion task to the
corresponding identity. Alternatively grant
appropriate schema-creation permissions when intentionally allowing resources
to create schemas. Do not grant prod schema access to the dev principal.

Verify serverless jobs, pipelines, and Connect availability in this workspace,
including policy, network, and billing permissions. The sample wheel task uses
serverless environment version `5`. Connect 16.4/Python 3.12 must be supported by
the serverless environment; validate compatibility when running the manual test.
This setup does not provision compute, execute data queries, or prove runtime
availability merely by validating bundle configuration.

## Manual integration testing

Only run this after authentication and serverless/sample-data access are ready.
It creates a real remote Spark session and can incur Databricks costs:

```powershell
uv sync --frozen --dev
$env:DATABRICKS_CONFIG_PROFILE = "nyc-taxi"
$env:DATABRICKS_SERVERLESS_COMPUTE_ID = "auto"
uv run --frozen pytest tests\integration -m integration
```

The fixture validates Connect compatibility and surfaces authentication/compute
errors. There is no silent compute fallback or skip-on-connection-failure.
Alternatively manually dispatch **Manual integration tests** from `main`.

Running the full sample job refreshes the ETL pipeline and can create tables
and incur additional costs. Run it only intentionally after deployment:

```powershell
databricks bundle run sample_job --target dev --profile nyc-taxi
```

## Troubleshooting

- **Missing client ID / bundle variable:** provision the principal and set
  environment variables; do not approve validation until setup is complete.
- **OIDC exchange failure:** check account audience, exact immutable subject,
  environment name, principal assignment, and job `id-token: write` permission.
- **Catalog/schema access error:** verify workspace binding and grants for the
  deploying and run identities, not just the local user's access.
- **Waiting for approval:** review the PR or main revision and approve the
  intended environment in GitHub. Fork PRs only need offline checks.
- **Non-main dispatch:** select `main`; neither production nor integration
  accepts an arbitrary feature branch.
- **Connect/serverless error:** check serverless availability and the Connect,
  Python, and environment compatibility before changing dependency versions.

Official template documentation:
https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/templates
