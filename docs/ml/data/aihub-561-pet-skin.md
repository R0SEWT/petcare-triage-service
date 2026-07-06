# AI-Hub 561 pet skin dataset access notes

Task: `petcare-triage-service-sg9`
Official source: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=561
AI-Hub dataset id: `561`
Korean title: `반려동물 피부 질환 데이터`
English title: no official English title found; "Companion Animal Skin Disease
Data" is an inferred translation.
Status: investigated, not acquired.

## Confirmed official facts

The official AI-Hub page describes a 2021 image dataset for companion animal
skin disease diagnosis, updated in 2024-04. The page states that only Korean
nationals can apply for the data (`내국인만 데이터 신청이 가능합니다.`).

Scale and scope:

- more than 10,000 companion animals
- dog and cat skin disease images
- more than 500,000 total images
- more than 250,000 disease images
- dog: 7 skin disease/symptom categories
- cat: 4 skin disease/symptom categories

Data format:

- source images: `jpg`
- medical-record/raw metadata: `csv`
- training labels: `json`
- labeling types: bounding box and segmentation
- image source: veterinary hospitals
- reported resolution in schema examples: `1920x1080`

## Official class vocabulary

General-camera / smartphone labels use `lesions` codes:

| Code | Korean label | Working English description | PetCare use |
| --- | --- | --- | --- |
| `A1` | `구진/플라크` | papule/plaque | candidate symptom feature, not a direct diagnosis |
| `A2` | `비듬/각질/상피성잔고리` | dandruff/scale/epidermal collarette | candidate symptom feature |
| `A3` | `태선화/과다색소침착` | lichenification/hyperpigmentation | candidate symptom feature |
| `A4` | `농포/여드름` | pustule/acne | candidate symptom feature |
| `A5` | `미란/궤양` | erosion/ulcer | candidate symptom feature |
| `A6` | `결절/종괴` | nodule/mass | candidate symptom feature; likely OOD for PetCare derm classifier |
| `A7` | `무증상` | asymptomatic/normal | `oodClass=healthy_skin` candidate |

Cytology labels use `diagnosis` codes:

| Code | Korean label | Working English description | PetCare use |
| --- | --- | --- | --- |
| `C1` | `감염성 피부염` | infectious dermatitis | possible broad diagnostic signal |
| `C6` | `비감염성 피부염` | non-infectious dermatitis | possible broad diagnostic signal |

Important mapping note: AI-Hub labels are mostly symptom morphology labels, not
the same as PetCare canonical diagnostic labels. They should not be mapped
directly to `atopic_dermatitis`, `bacterial_pyoderma`, `dermatophytosis`, etc.
without a separate adjudication rule.

## Official counts

The official page exposes these image counts.

Dog:

| Category | Symptomatic | Asymptomatic |
| --- | ---: | ---: |
| `A1` papule/plaque | 40,600 | 40,600 |
| `A2` dandruff/scale/epidermal collarette | 67,300 | 67,300 |
| `A3` lichenification/hyperpigmentation | 67,300 | 67,300 |
| `A4` pustule/acne | 15,000 | 15,000 |
| `A5` erosion/ulcer | 15,000 | 15,000 |
| `A6` nodule/mass | 15,000 | 15,000 |

Dog cytology:

| Category | Image count |
| --- | ---: |
| `C1` infectious dermatitis | 3,000 |
| `C6` non-infectious dermatitis | 3,000 |

Cat:

| Category | Symptomatic | Asymptomatic |
| --- | ---: | ---: |
| `A4` pustule/acne | 8,460 | 8,460 |
| `A2` dandruff/scale/epidermal collarette | 8,460 | 8,460 |
| `A6` nodule/mass | 8,460 | 8,460 |

Cat cytology:

| Category | Image count |
| --- | ---: |
| `C1` infectious dermatitis | 1,420 |
| `C6` non-infectious dermatitis | 1,420 |

## Access and redistribution

Confirmed from AI-Hub policy and the dataset page:

- a logged-in AI-Hub account and dataset download approval are required
- API downloads use `aihubshell`, an API key, and the dataset key after approval
- users must state purpose and pass applicant verification
- foreign individuals/entities and offshore export require separate agreement
  with the operating/participating institutions and NIA
- AI-Hub data must be attributed to NIA in derivative works
- AI-Hub data may not be shown, provided, transferred, lent, or sold to another
  party without approval
- sale or commercial use of the dataset itself requires separate agreement
- if personal information is found, AI-Hub must be notified and the downloaded
  data must be deleted

Practical PetCare decision: do not download, mirror, publish to Hugging Face, or
train on this dataset until formal access and redistribution/use terms are
approved for this project. If access is approved, keep the raw data private and
record approval terms with the manifest.

## Official file inventory

The public file-list endpoint returned 6 ZIP files under
`152.반려동물 피부질환 데이터/01.데이터`, total size `410,153,333,964` bytes
(`382 GiB`). This is an inventory only; no dataset files were downloaded.

| Split | Kind | File | File key | Bytes |
| --- | --- | --- | ---: | ---: |
| Training | source data | `TS01.zip` | `517017` | 96,715,412,955 |
| Training | source data | `TS02.zip` | `517018` | 85,477,514,695 |
| Training | labeling data | `TL01.zip` | `517019` | 96,715,763,529 |
| Training | labeling data | `TL02.zip` | `517020` | 86,016,937,297 |
| Validation | source data | `VS01.zip` | `517021` | 22,575,819,068 |
| Validation | labeling data | `VL01.zip` | `517022` | 22,651,886,420 |

Verification command:

```bash
curl -L -sS -X POST -d 'dataSetSn=561' \
  'https://aihub.or.kr/aihubdata/data/aJaxS3FileList.do'
```

## Secondary references and mirror risk

Secondary references confirm that teams have used this dataset in competitions
and projects, especially with dog-only `A1`-`A6` or `A1`-`A7` classifiers:

- https://aifactory.space/task/2347/overview
- https://aifactory.space/task/2157/overview
- https://github.com/gladuz/pet-disease-recognition
- https://github.com/Jihwan98/KT-AIVLE-BigProject
- https://github.com/aengzu/AILectureProject

Several Roboflow datasets appear likely to be partial derivatives because they
use Korean class names or numeric `A*`-style mappings. Treat them as untrusted
mirrors until provenance, license, and deduplication are proven:

- https://universe.roboflow.com/rr-s1o7d/petskin123456
- https://universe.roboflow.com/a4-fstay/a5-thpbo
- https://universe.roboflow.com/a2-eerh2/a5-a6

Risk: a derivative can claim a permissive license while the upstream AI-Hub
terms still prohibit redistribution or third-party transfer. Do not import these
mirrors into PetCare without source verification and license review.

## Current decision

AI-Hub 561 is a high-value candidate for future private training, especially for
symptom segmentation, OOD/healthy skin, and morphology pretraining. It is not an
immediate downloadable training source for this project because access,
nationality, foreign-use, and redistribution constraints are unresolved.

Next valid actions:

- apply through AI-Hub only if a Korean applicant/partner and project terms are
  available
- ask AI-Hub/NIA whether PetCare can use the data from Peru/US-based tooling and
  whether model weights trained on it may be exported or published
- avoid Roboflow/GitHub mirrors unless provenance and upstream permission are
  confirmed
- if approved, acquire privately, preserve all raw ZIPs, generate a manifest, and
  keep it out of public HF/Git storage
