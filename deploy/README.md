# GCP Cloud Run deployment

This project targets **Google Cloud Run** because the API is already packaged
as a stateless container, Cloud Run provides managed HTTPS and scale-to-zero,
and Google Cloud Storage supplies a simple startup model artifact source. The
deployment remains manual; CI never deploys.

## Prerequisites

Install the Google Cloud CLI and Docker, create a fresh GCP project with
billing enabled, and authenticate locally with an account allowed to create
Artifact Registry repositories, buckets, service accounts, and Cloud Run
services. Copy `deploy/.env.example` to a private local file and replace its
placeholders. Never commit that file or credentials.

## 1. Set variables locally

PowerShell example (values are placeholders):

```powershell
$env:GCP_PROJECT_ID = "your-project"
$env:GCP_REGION = "us-central1"
$env:GCS_BUCKET = "your-unique-model-bucket"
$env:SERVICE_NAME = "constituency-insights-api"
$env:ARTIFACT_REPOSITORY = "constituency-insights"
```

## 2. Prepare the model artifact

Create a gzip tar archive containing the complete frozen stance checkpoint
directory (`model.safetensors`, tokenizer files, and `config.json`). Compute its
SHA-256 locally and upload it to the private bucket. Also record the
`model.safetensors` SHA-256; it must equal the hash in
`configs/serving.docker.yaml` and `/model-info`.

```text
tar -czf stance-distilbert.tar.gz -C outputs/local/stance_training/finetuning/distilbert_full best_checkpoint
sha256sum stance-distilbert.tar.gz
gcloud storage cp stance-distilbert.tar.gz gs://YOUR_BUCKET/checkpoints/stance-distilbert.tar.gz
```

Do not run these commands until the bucket, object name, and hashes have been
reviewed. The archive is private model data; do not make the bucket public.

## 3. Create infrastructure and deploy manually

The following commands are intentionally not run by this repository:

```text
gcloud config set project "$GCP_PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com storage.googleapis.com
gcloud artifacts repositories create "$ARTIFACT_REPOSITORY" --repository-format=docker --location="$GCP_REGION"
gcloud iam service-accounts create constituency-insights-runtime
gcloud storage buckets create "gs://$GCS_BUCKET" --location="$GCP_REGION" --uniform-bucket-level-access
gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" --member="serviceAccount:constituency-insights-runtime@$GCP_PROJECT_ID.iam.gserviceaccount.com" --role=roles/storage.objectViewer
gcloud auth configure-docker "$GCP_REGION-docker.pkg.dev"
docker build -t "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPOSITORY/api:latest" .
docker push "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPOSITORY/api:latest"
gcloud run deploy "$SERVICE_NAME" --image="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$ARTIFACT_REPOSITORY/api:latest" --region="$GCP_REGION" --service-account="constituency-insights-runtime@$GCP_PROJECT_ID.iam.gserviceaccount.com" --set-env-vars="MODEL_GCS_BUCKET=$GCS_BUCKET,MODEL_GCS_OBJECT=checkpoints/stance-distilbert.tar.gz,MODEL_GCS_ARCHIVE_SHA256=REPLACE_ARCHIVE_HASH,OUTCOME_GCS_OBJECT=artifacts/outcome_model.joblib,OUTCOME_MODEL_SHA256=REPLACE_OUTCOME_HASH,OUTCOME_CONFORMAL_OBJECT=artifacts/conformal.json,OUTCOME_CONFORMAL_SHA256=REPLACE_CONFORMAL_HASH" --cpu=2 --memory=4Gi --max-instances=3 --no-allow-unauthenticated
```

Grant invocation only to the intended caller. Cloud Run startup downloads the
model archive, rejects path traversal, verifies both the archive hash (when
supplied) and the frozen weights hash, and only then loads the model.

## 4. Verify deployment

Obtain the service URL with `gcloud run services describe` and run:

```text
python scripts/smoke_test_api.py --base-url https://YOUR_SERVICE_URL
```

The smoke test checks `/health` and sends a placeholder stance request. It does
not print request text. Check `/model-info` and confirm the checkpoint hash and
training date before accepting traffic.

## Rollback and updates

Publish a new immutable image tag and model archive, update the Cloud Run
revision manually, and repeat the hash check. Never overwrite a production
archive in place.
