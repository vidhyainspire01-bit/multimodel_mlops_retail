# ecommerce-mlops

Databricks MLOps demo on Azure: H&M data, two models (repurchase classifier, weekly demand
time series), Databricks Asset Bundles, GitHub Actions promoting dev -> staging -> prod.

One workspace, three Unity Catalog catalogs: `dev_ml`, `stage_ml`, `prod_ml`.

## Flow
| Trigger | Workflow | Does |
|---|---|---|
| Pull request | `pr-validation` | ruff, pytest, deploy to dev, run ingest + pipeline |
| Merge to `main` | `deploy` / staging | deploy to `stage_ml`, run integration pipeline (gate enforced) |
| Staging green + approval | `deploy` / prod | deploy to `prod_ml`, smoke run, Monday schedule unpaused |

## One-time setup
1. Create catalogs `dev_ml`, `stage_ml`, `prod_ml` (Catalog Explorer).
2. Create a Databricks service principal + OAuth secret; grant it `USE CATALOG`,
   `CREATE SCHEMA`, `CREATE TABLE`, `CREATE MODEL` on the three catalogs, and
   `READ VOLUME` on `dev_ml.hm.raw`.
3. In GitHub -> Settings -> Environments create `dev`, `staging`, `production`. Add secrets
   `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` to each.
   Add required reviewers to `production`.
4. Link GitHub in Databricks and create a Git folder for this repo.

## Layout
- `databricks.yml` - bundle, variables, dev/staging/prod targets
- `resources/` - job definitions (ingest, repurchase pipeline)
- `notebooks/` - notebooks as `.py` source files (shared code imported from `common/`)
- `common/` - promotion gate (`promote_if_better`)
- `tests/` - pytest unit tests run on every PR
