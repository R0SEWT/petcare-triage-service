# Roboflow dog dermatology candidates

Task: `petcare-triage-service-el9`
Status: evaluated, not downloaded.

## Short decision

Roboflow is useful for a fast PetCare bootstrap, but only as noisy training data
and model smoke-test infrastructure. It must not be treated as a credible eval
source. The best next step is to run hosted/local Roboflow Inference against a
small fixed PetCare probe set, then export the most useful dataset versions only
if the model behavior looks plausible.

## Best candidate for our current taxonomy

Primary candidate:

- URL: https://universe.roboflow.com/dog-skin-disease-dermatosis/dog-skin-disease-dataset
- Task: classification
- License shown by Roboflow: CC BY 4.0
- Size shown on project page: 1,418 images
- Versions: 2
- Models: 2
- Reported model accuracy: 94.8%
- Current model id: `dog-skin-disease-dataset/2`
- Model type: ViT Classification
- Classes:
  - `healthy`
  - `bacterial dermatosis`
  - `fungal infection`
  - `hypersensitivity dermatitis`
  - `Unlabeled`

This is the closest match to the Mendeley class vocabulary and our current
`yolov8-cls` baseline path. The model page also reports a backing dataset size
of 4,398 images for version 2, which does not match the project-level 1,418
image count; verify by export metadata before training.

Working PetCare mapping:

| Roboflow class | PetCare mapping | Use |
| --- | --- | --- |
| `healthy` | `condition=unknown`, `oodClass=healthy_skin` | OOD/negative |
| `bacterial dermatosis` | `condition=bacterial_pyoderma`, `oodClass=in_scope` | noisy train |
| `fungal infection` | `condition=fungal_malassezia`, `oodClass=in_scope` | noisy train; verify whether ringworm vs Malassezia |
| `hypersensitivity dermatitis` | `condition=atopic_dermatitis`, `oodClass=in_scope` | noisy train; broad allergy label |
| `Unlabeled` | `condition=unknown`, `oodClass=unlabeled_source` | preserve but exclude from supervised train |

## Other useful candidates

### Dog skin disease prediction

- URL: https://universe.roboflow.com/majorproject-kopqr/dog-skin-disease-prediction
- Task: classification
- License shown by Roboflow: CC BY 4.0
- Images: 361
- Versions/models: 3
- Reported accuracy: 95.8%
- Model id: `dog-skin-disease-prediction/3`
- Model type: ViT Multi-label Classification
- Classes: `flea_allergy`, `hotspot`, `mange`, `ringworm`

Use as a diagnostic probe only. It is more clinically named than the primary
candidate, but the sample count is small and the taxonomy does not align cleanly
with PetCare canonical labels.

### Dog skin disease detection

- URL: https://universe.roboflow.com/myprojects-zsnac/dog-skin-disease-detection-6pgvk
- Task: classification
- License shown by Roboflow: CC BY 4.0
- Images: 359
- Versions/models: 1
- Reported accuracy: 96.9%
- Model id: `dog-skin-disease-detection-6pgvk/1`
- Model type: Roboflow 2.0 Multi-label Classification
- Classes: `flea_allergy`, `hotspot`, `mange`, `ringworm`, `Unlabeled`

Use as a small smoke-test model only. It is too small and taxonomy-shifted to be
the main training source.

### Litespy dog skin diseases

- URL: https://universe.roboflow.com/litespy-l22hu/dog-skin-diseases
- Task: object detection
- License shown by Roboflow: CC BY 4.0
- Images: 618
- Model id: `dog-skin-diseases/1`
- Model type: Roboflow 3.0 Object Detection Fast
- Reported metrics: mAP@50 39.3%, precision 25.1%, recall 99.9%
- Classes: `healthy`, `bacterial-dermatosis`, `dog-skin-diseases`,
  `fungal-infection`, `hypersensitivity-allergic-dermatosis`

This has useful localization labels, but the reported precision is weak. Use
only if we add a skin-region detector or cropper; do not use the metric as model
quality evidence.

### Gian dog skin diseases

- URL: https://universe.roboflow.com/gian-ocdnp/dog-skin-diseases-b7b5y-jvz7i
- Task: object detection
- License shown by Roboflow: CC BY 4.0
- Images: 1,379
- Model id: `dog-skin-diseases-b7b5y-jvz7i/1`
- Model type: Roboflow 3.0 Object Detection Fast
- Reported metrics: mAP@50 91.7%, precision 91.5%, recall 85.2%
- Classes: `Demodicosis`, `Ringworm`

This is useful for out-of-scope/ringworm probing, but not a direct fit for the
current PetCare 5-class classifier.

## Pretrained-model reality

Confirmed from Roboflow docs:

- Universe models can be run by `model_id` through Roboflow Inference.
- Hosted API and local Inference both require a Roboflow API key for Universe
  or fine-tuned project models.
- Local `inference.get_model(model_id=...)` fetches/caches model weights
  automatically for inference.
- Manual raw weight download is a paid/Core-or-Enterprise feature and depends on
  model compatibility.

Implication: there are already trained models we can use immediately for
inference/smoke testing, but they are not ideal as open checkpoints for local
fine-tuning. For controlled training, export the dataset and fine-tune our own
model from a known base checkpoint (`yolov8n-cls`, `efficientnet`, `dino/vit`,
etc.).

## Recommended PetCare path

1. Run a zero-download smoke test through Roboflow Inference:
   - `dog-skin-disease-dataset/2` for taxonomy-aligned classification
   - `dog-skin-disease-prediction/3` as a second opinion on common conditions
   - optional: `dog-skin-diseases/1` for localization/crop behavior
2. Compare outputs on:
   - Mendeley labeled images
   - Mendeley `unlabeled_source` images
   - obvious non-skin dog/cat images
   - human skin OOD examples
3. If the taxonomy-aligned model is plausible, export only the project/version
   metadata and a dataset copy for training. Keep raw and normalized manifests.
4. Run perceptual dedup against Mendeley and any future gold set before using
   the images.
5. Train our own lightweight baseline locally/cloud-side from exported data. Do
   not report Roboflow page accuracy as PetCare performance.

## Risk notes

- Roboflow project licenses are self-declared by uploaders. They do not prove
  that upstream source images were legally redistributable.
- Some Roboflow projects may be partial derivatives of AI-Hub or Kaggle/Mendeley
  data. Treat them as training-only until provenance and dedup are checked.
- The "Unlabeled" class should be preserved in manifests but excluded from
  supervised condition training.
- Reported Universe metrics are internal to the uploaded split; they are not
  comparable to a vet-verified PetCare gold set.

## No-download verification sources

- https://universe.roboflow.com/dog-skin-disease-dermatosis/dog-skin-disease-dataset
- https://universe.roboflow.com/majorproject-kopqr/dog-skin-disease-prediction
- https://universe.roboflow.com/myprojects-zsnac/dog-skin-disease-detection-6pgvk
- https://universe.roboflow.com/litespy-l22hu/dog-skin-diseases
- https://universe.roboflow.com/gian-ocdnp/dog-skin-diseases-b7b5y-jvz7i
- https://inference.roboflow.com/quickstart/run_a_model/
- https://inference.roboflow.com/quickstart/load_from_universe/
- https://docs.roboflow.com/deploy/download-roboflow-model-weights
