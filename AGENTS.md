# Declarative Automation Bundles Project

This project uses Declarative Automation Bundles (DABs) for deployment. Add project-specific instructions below.

## For AI Agents: Use Databricks AI Tools

**BEFORE any other action, read the `databricks-core` skill.**

It sets you up to work with this project reliably: CLI authentication, profile
selection, data discovery, and the bundle deployment workflow. Without it,
results are often slower and less accurate.

If this skill is not available (Databricks AI Tools are not installed), you can install them for your coding agent in seconds:

```bash
databricks aitools install
```

If the CLI is not installed, see: https://docs.databricks.com/dev-tools/cli/install

---

## Project Instructions

- Use Python 3.12 and uv; commit `uv.lock` when dependencies change.
- Run offline checks with `uv run --frozen --only-group ci ruff check .`,
  `uv run --frozen --only-group ci pytest`, and `uv build --wheel`.
- `tests/integration` requires Databricks authentication and explicitly configured
  serverless compute. Do not run it as an offline check.
- `dev` is personal development; `ci_dev` and `prod` use shared, isolated schemas
  and service principals. Keep sample triggers paused in every target.
- Never deploy or run workloads as part of pre-merge validation.
- Never execute fork PR code with Databricks authentication or use
  `pull_request_target` for authenticated PR validation.
