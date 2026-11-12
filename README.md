# PosterLlama + HyperCLOVA X

이미지와 홍보 문구를 입력해 HTML 형식의 포스터 레이아웃을 생성하는 2단계 학습 구현입니다. `~/Desktop/car/PosterLlama`의 논문 코드를 참고했습니다. 실행 진입점은 **[test.ipynb](test.ipynb)** 입니다.

## 빠른 실행

Python 3.10+ 커널에서 노트북을 위에서부터 실행하세요. 기본 `MODE = "smoke"`는 가중치 다운로드 없이 작은 무작위 SigLIP/Llama 모델로 두 단계 학습, PEFT LoRA, 저장·복원 및 생성을 실행합니다. **이 모드는 구조 검증용이며 학습된 HyperCLOVA 모델이나 유효한 포스터 생성 결과가 아닙니다.**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name posterllama-hcx --display-name "PosterLlama HCX"
jupyter lab test.ipynb
```

VS Code에서는 설치한 환경을 노트북 커널로 선택해도 됩니다. 현재 개발 머신에서는 `/opt/homebrew/anaconda3/envs/naver/bin/python` 환경으로 검증했습니다. 기본 `python3`가 Python 3.7을 가리킬 수 있으므로 커널 버전을 확인하세요.

## 아키텍처

```text
image → frozen HyperCLOVA SigLIP vision tower → patch grouping → linear Adapter
                                                                 ↓ visual tokens
task / text constraints → CodeLlama tokenizer + embeddings → frozen CodeLlama (+ LoRA)
                                                                 ↓
                                              stage 1: caption / stage 2: HTML
```

| 단계 | 비전 타워 | Adapter | CodeLlama 기본 가중치 | LoRA | 정답 |
|---|---|---|---|---|---|
| 1 | 동결 | 학습 | 동결 | 없음 | 이미지 설명 |
| 2 | 동결 | 1단계 가중치 동결 | 동결 | 학습 | SVG rect 좌표를 포함한 HTML |

- 그림의 “HyperCLOVA X CLIP”은 체크포인트가 지정되지 않아 공개 **`naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B`의 SigLIP 비전 타워**로 구현했습니다. 독립적인 CLIP 제품이나 HyperCLOVA API 호출이 아닙니다. CodeLlama는 `codellama/CodeLlama-7b-hf`를 유지했습니다.
- 공식 모델의 `vision_model.*` 가중치만 로드합니다. HyperCLOVA의 언어 모델과 C-Abstractor는 사용하지 않으며 `trust_remote_code`도 필요 없습니다. 비전 가중치가 포함된 shard 전체(현재 약 5 GB)는 처음 한 번 다운로드됩니다.
- 원본 `minigpt4.py`처럼 연속 패치 4개를 이어 붙이고 선형 투영합니다. SigLIP에는 CLS 토큰이 없으므로 첫 패치를 제거하지 않습니다. 729개 패치를 732개로 zero-pad하여 183 visual tokens를 생성합니다. 공식 HCX 설정과 같이 `hidden_states[-2]`를 사용합니다.
- 입력은 378×378 한 장으로 letterbox 처리하며 평균/표준편차 0.5로 정규화합니다. 공개 HCX 전체 VLM의 AnyRes 다중 crop 경로는 사용하지 않습니다. 이것은 PosterLlama에 비전 타워를 이식한 변형이며 전체 HCX VLM 재현이 아닙니다.
- 텍스트 토큰은 LLM embedding에서 생성합니다. Adapter는 visual tokens만 생성합니다. 이미지·프롬프트·padding은 loss에서 제외하고 정답+EOS만 학습합니다. 1단계에서도 frozen LLM을 통과하는 gradient는 유지합니다.
- 기존 DINO 기반 Adapter는 입력 차원과 특징 분포가 달라 재사용할 수 없습니다. HCX로 1단계를 새로 학습하세요.

## 실제 학습

1. 노트북에서 `MODE = "real"`로 바꾸고 1·2단계 학습 JSONL 및 이미지 경로를 지정합니다.
2. `ModelConfig`에서 모델 ID 또는 로컬 모델 경로를 지정합니다. 재현이 필요하면 `vision_revision`, `language_revision`에 고정 commit을 사용하세요.
3. 1단계 학습 후 `adapter.safetensors`를 저장합니다.
4. 저장한 Adapter를 읽어 동결한 뒤 2단계 LoRA를 학습합니다.
5. `PosterLlama.from_checkpoint(...)`로 복원하고 `generate(...)`를 실행합니다.

실학습에는 CUDA GPU 사용을 권장합니다. 메모리 사용량은 배치·시퀀스·정밀도에 따라 달라집니다. `gradient_checkpointing=True`, 배치 1, gradient accumulation을 기본 제공하며, CUDA에서 `bitsandbytes` 설치 후 `load_in_4bit=True`로 설정할 수 있습니다. 4-bit는 양자화된 기본 모델에 Adapter/LoRA를 학습하는 옵션으로, 비양자화 학습과 결과가 동일하다고 보장하지 않습니다. CPU/Mac에서는 우선 smoke 모드를 사용하세요.

기본 `dtype="auto"`는 지원되는 CUDA에서 bf16, 그 외 float32입니다. 노트북의 실학습 에폭/학습률은 시작용 예시이며 논문 결과를 보장하는 설정이 아닙니다. 별도 검증 데이터로 loss 및 레이아웃 품질을 평가하세요. 논문 데이터나 대형 모델 가중치는 저장소에 포함되어 있지 않습니다.

## 데이터

UTF-8 JSONL 한 줄에 한 샘플을 기록합니다. `image`는 `image_root` 기준이며, 지정하지 않으면 manifest 파일의 부모 폴더 기준입니다. JSON 배열 파일도 지원합니다.

1단계:

```json
{"image":"images/chair.jpg","caption":"오른쪽에 컵홀더가 있는 흰색 리클라이너 의자입니다."}
```

2단계:

```json
{"image":"images/chair.jpg","prompt":"512x360 포스터에 신제품 출시 문구와 로고를 배치하세요. HTML을 출력하세요.","html":"<html><body><svg width=\"512\" height=\"360\"><rect data-category=\"text\" x=\"20\" y=\"20\" width=\"200\" height=\"50\" /></svg></body></html>"}
```

`build_layout_prompt(texts, width, height)`로 작업 정의·텍스트 조건·HTML 형식을 포함한 프롬프트를 만들 수 있습니다. 생성된 좌표는 letterbox 이미지가 아닌 원본 캔버스 기준입니다.

원본 2단계 데이터의 `name`, `cond_cate_to_size_pos_seq_modeling`, `unconditional_seq_modeling` 등 7개 작업 필드와 `<MID>` 구분자를 지원합니다. `task="random"`은 원본 가중치 비율로 작업을 선택합니다. `augment=True`이면 `name` 목록에서 이미지를 샘플링합니다. 원본 1단계 WebDataset tar 스트리밍은 포함하지 않았으므로 이미지/캡션 JSONL로 준비해야 합니다.

입력 또는 정답이 토큰 제한을 초과하면 오류를 발생시킵니다. HTML을 잘라 잘못된 정답으로 학습하지 않습니다. 필요하면 `max_prompt_tokens`, `max_target_tokens`를 모델 컨텍스트 한도 안에서 조정하세요.

## 체크포인트와 결과

```text
outputs/hcx_codellama/stage1/
  posterllama.json        # 구조 및 모델 ID/revision
  adapter.safetensors
  history.json
outputs/hcx_codellama/stage2/
  posterllama.json
  adapter.safetensors
  lora/                  # PEFT adapter_config.json / adapter_model.safetensors
  history.json
```

기본 LLM/비전 가중치는 포함하지 않으며 복원 시 같은 모델을 다시 로드합니다. 체크포인트는 추론/가중치 재사용용이며 optimizer·scheduler·RNG까지 복구하는 학습 재개 기능은 아닙니다. 같은 출력 폴더에서 재학습하면 해당 체크포인트가 갱신됩니다.

`render_layout(image, html)`은 생성 HTML을 실행하지 않고 좌표만 읽어 박스를 그립니다. 완성된 광고 문구의 타이포그래피나 로고 이미지를 합성하는 도구는 아닙니다. 잘린 SVG, 숫자가 아닌 값, 캔버스 밖 박스는 오류로 표시합니다. HTML의 문법과 배치 품질은 모델 학습량에 따라 달라집니다.

## 코드와 검증

- `posterllama_hcx/vision.py`: 공식 checkpoint에서 비전 타워만 추출, 전처리, Adapter
- `posterllama_hcx/model.py`: 멀티모달 입력, 단계별 동결, 생성, checkpoint
- `posterllama_hcx/data.py`: JSONL/원본 논문 포맷, 프롬프트
- `posterllama_hcx/training.py`: token 수 기준 gradient accumulation, validation, 저장
- `posterllama_hcx/layout.py`: 좌표 파싱·미리보기
- `posterllama_hcx/smoke.py`: 다운로드 없는 소형 실제 모델 및 합성 데이터

```bash
python -m pytest -q tests
```

테스트는 실제 소형 SigLIP/Llama/PEFT를 사용해 두 단계 업데이트, 동결 유지, gradient 전달, EOS/padding 마스킹, 729 패치 처리, HCX 가중치 이름 매핑, 체크포인트 복원, accumulation 및 레이아웃 검증을 확인합니다. 대형 HCX/CodeLlama 가중치 학습이나 실제 데이터 성능 평가는 별도 실행이 필요합니다.

참고: [HCX 공식 모델](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B), [공식 설정](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B/blob/main/config.json), [공식 비전 forward 구현](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B/blob/main/modeling_hyperclovax.py). 로컬 원본의 `src/model/minigpt4.py`, `src/model/minigpt_base.py`, `src/dataset/custom_dataset.py`, `src/common/configs_stage*_dino_codellama.py`를 구조 참고 자료로 사용했습니다.
