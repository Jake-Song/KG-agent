# AI4Sci Molecule

ESOL을 시작점으로 분자 표현과 용해도 회귀를 익히는 프로젝트입니다. 1주차에는
RDKit descriptor baseline을 만들고, 2주차에는 atom embedding과 message passing으로
graph representation을 학습하는 GCN/GIN을 같은 random split에서 비교합니다.
3주차에는 같은 모델을 random split과 Murcko scaffold split에서 seed 5개로 다시
평가해, GNN이 물리적 관계를 배웠는지 아니면 익숙한 molecular motif에 기대고 있는지를
확인합니다. 4주차에는 새로 학습하지 않고 3주차의 분자 단위 예측을 13개 molecular property와
대조해, prediction error를 데이터에 대한 진단 도구로 씁니다. 5주차에는 seed만 바꾼 deep
ensemble로 예측에 uncertainty를 붙이고, 그 uncertainty가 오차의 순위를 맞히는지·확률적으로
보정되어 있는지, 그리고 그 보정이 낯선 scaffold까지 버티는지를 무학습 similarity 대조군과
비교합니다. 6주차에는 그 uncertainty를 소비합니다. 측정된 분자 100개에서 출발해 나머지
pool에서 다음에 실험할 분자를 모델이 고르게 하고, 무작위로 고를 때보다 label을 몇 개
아끼는지를 in-domain과 out-of-scaffold 두 regime에서 잽니다. 무학습 diversity 대조군과
정답을 훔쳐보는 oracle 상한선이 곡선을 아래위로 끼워, 이득이 모델의 앎에서 왔는지 그저
화학공간을 넓힌 덕인지 구분합니다.

## 설치

Python 3.13과 [uv](https://docs.astral.sh/uv/)가 필요합니다.
기본 환경은 PyTorch의 공식 CUDA 13 wheel을 사용합니다. CUDA 13을 지원하는
NVIDIA 드라이버가 설치된 Linux/WSL x86-64 환경에서 다음 명령을 실행합니다.

```bash
uv sync
```

GPU가 없는 환경에서는 `gpu` 그룹을 끄고 CPU-only wheel을 설치합니다.

```bash
uv sync --no-group gpu --group cpu
```

CPU 환경에서 명령을 실행할 때도 같은 그룹 선택을 지정합니다.

```bash
uv run --no-group gpu --group cpu <command>
```

첫 데이터 로드 시 PyG가 ESOL 원본을 내려받아 `data/MoleculeNet/`에 저장합니다.
`data/`는 `.gitignore`에 포함되어 있어 커밋되지 않습니다.

## 노트북 실행

노트북은 다음 일곱 개입니다.

- [`notebooks/01_molecule_exploration.ipynb`](notebooks/01_molecule_exploration.ipynb):
  descriptor 탐색과 Linear/MLP baseline
- [`notebooks/02_gnn_solubility.ipynb`](notebooks/02_gnn_solubility.ipynb):
  GCN/GIN 학습과 descriptor baseline 비교
- [`notebooks/03_scaffold_split.ipynb`](notebooks/03_scaffold_split.ipynb):
  random split과 scaffold split의 generalization gap 비교
- [`notebooks/04_error_analysis.ipynb`](notebooks/04_error_analysis.ipynb):
  prediction error와 molecular property의 관계, 모델 간 오차 일치도
- [`notebooks/05_uncertainty.ipynb`](notebooks/05_uncertainty.ipynb):
  deep ensemble uncertainty의 순위·selective prediction·calibration, 그리고 novelty별 전이
- [`notebooks/06_active_learning.ipynb`](notebooks/06_active_learning.ipynb):
  active learning의 학습곡선·예산 절약·cold start·무엇을 사는가
- [`notebooks/06_colab_runner.ipynb`](notebooks/06_colab_runner.ipynb):
  Colab GPU에서 6주차 sweep을 실행하고 끊긴 자리에서 재개하는 러너

노트북을 처음부터 다시 실행하려면 다음 명령을 사용합니다.

```bash
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=1800 \
  notebooks/02_gnn_solubility.ipynb notebooks/03_scaffold_split.ipynb \
  notebooks/04_error_analysis.ipynb notebooks/05_uncertainty.ipynb \
  notebooks/06_active_learning.ipynb
```

`06_colab_runner.ipynb`는 Colab 전용이므로 여기에 넣지 않습니다.

3주차 노트북은 `results/week3/`에 저장된 sweep 결과를 재사용하므로 보통 몇 초 만에
끝납니다. 캐시를 지우고 다시 계산하면 GPU에서 약 10분이 걸리므로 타임아웃을 넉넉히
잡아 둡니다. 4주차 노트북은 학습을 하지 않고 3주차 예측만 읽으므로 CPU에서 1분 이내에
끝납니다.

대화형으로 살펴보려면 `uv run jupyter lab`을 실행합니다.

## 결과물

3주차 sweep의 표와 그림은 `results/week3/`에 커밋되어 있습니다
(`metrics.csv`, `summary.json`, `splits.json`, `figures/*.png`). 분자 단위 예측값인
`predictions.csv`는 1 MB에 가까워 `.gitignore` 대상이며, 아래 명령으로 나머지 결과와
함께 다시 만들 수 있습니다(GPU 기준 약 10분).

```bash
uv run python -m ai4sci_molecule.week3
```

4주차 분석의 표와 그림은 `results/week4/`에 커밋되어 있습니다. 분자 단위
`molecule_errors.csv`는 1.1 MB로 `.gitignore` 대상이며, 아래 명령으로 나머지 결과와 함께 다시
만듭니다. 학습이 없으므로 3주차 캐시가 있으면 CPU에서 1분 이내에 끝납니다.

```bash
uv run python -m ai4sci_molecule.week4
```

5주차 앙상블 스윕의 표와 그림은 `results/week5/`에 커밋되어 있습니다. 멤버 단위
`member_predictions.csv`는 20,325행으로 `.gitignore` 대상이며, 아래 명령으로 나머지 결과와 함께
다시 만듭니다. 이번 주는 새로 학습하므로(GNN 60회) GPU에서 약 20~25분이 걸립니다.

```bash
uv run python -m ai4sci_molecule.week5
```

6주차 active learning sweep의 표와 그림은 `results/week6/`에 커밋되어 있습니다. 궤적 단위
shard(`trajectories/` 50개, `acquisitions/` 50개, `test_predictions/` 50개, `round0/` 10개)도
함께 커밋했으므로, 새로 clone한 저장소에서도 **재학습 없이** 표와 그림을 다시 만들 수
있습니다. 합쳐 놓은 `test_predictions.csv`만 `.gitignore` 대상입니다.

처음부터 학습하면 궤적 50개, GNN 1,230회입니다. 다만 라운드마다 학습 집합이 80~400개뿐이라
epoch당 미니배치가 2~7개에 그치고, early stopping도 대개 40~100 epoch에서 걸립니다. 이
저장소의 sweep은 **RTX 2080 SUPER에서 1시간 37분**(학습 1회 약 4.7초) 걸렸습니다. 같은
설정을 GPU 없는 CPU에서 재 보면 학습 1회 6.3초, 전체 약 2시간 10분입니다. 이 워크로드는
연산이 아니라 커널 런치와 PyG의 Python 콜레이션에 묶여 있어서, GPU를 붙여도 1.3배 남짓밖에
빨라지지 않습니다. GPU를 못 구했다고 못 돌릴 규모가 아닙니다. 아래 Colab 절도 참고하세요.

결과를 한 줄로 적으면, **모델이 고르게 해서 아낀 실험은 낯선 scaffold에서만 나왔고, 거기서
이긴 것은 앙상블 spread(`uncertainty`)가 아니라 화학공간 커버리지(`diversity`, label 500개
기준 1.71배)였습니다.** in-domain에서는 쓸 수 있는 어떤 전략도 무작위를 이기지 못했습니다.
자세한 판정 기준과 반례는 [`reports/week6.md`](reports/week6.md)에 있습니다.

```bash
uv run python -m ai4sci_molecule.week6
```

노트북과 `load_or_run_sweep` / `load_or_run_error_analysis` /
`load_or_run_uncertainty_study` / `load_or_run_active_learning_study`는 캐시 파일이 하나라도
없거나 설정이 다르면 자동으로 다시 계산하므로, 새로 clone한 저장소에서도 그대로 실행됩니다.
5주차 캐시는 한 걸음 더 나아가, 학습에 영향을 주지 않는 설정만 바뀌면(`bootstrap_samples` 등)
재학습 없이 표만 다시 만듭니다. 6주차는 캐시가 세 단계입니다: 저장된 표 → 궤적 shard →
실제 학습. 그래서 분석 설정만 바꾸면 shard에서 표만 다시 만들고, 궤적 하나가 없으면 그
궤적만 다시 학습합니다.

해석과 한계는 [`reports/week3.md`](reports/week3.md),
[`reports/week4.md`](reports/week4.md), [`reports/week5.md`](reports/week5.md),
[`reports/week6.md`](reports/week6.md)에 정리해 두었습니다.
`data/`도 다시 내려받을 수 있는 원본이므로 계속 `.gitignore` 대상입니다.

## Colab GPU에서 6주차 sweep 돌리기

로컬에 CUDA가 없으면 [`notebooks/06_colab_runner.ipynb`](notebooks/06_colab_runner.ipynb)를
Colab에서 엽니다. sweep이 궤적 단위로 저장되므로 세션이 끊겨도 **실행 셀만 다시 실행하면
남은 궤적부터 이어서** 돌아갑니다. 결과를 Google Drive에 두면 런타임이 완전히 초기화돼도
shard가 살아남습니다.

러너는 Colab에 이미 깔린 torch(CUDA 포함)를 그대로 쓰고 `torch-geometric`과 `rdkit`만
채운 뒤, 저장소를 `pip install -e . --no-deps`로 커널에 설치합니다. 노트북 커널에서 직접
import해야 대화형 재개 셀을 쓸 수 있기 때문입니다(`uv sync`로 만든 가상환경은 커널에서
import되지 않습니다). Colab의 파이썬이 3.12일 수 있어 `requires-python`은 `>=3.12`입니다.
로컬에서는 여전히 `uv.lock` 그대로 3.13 + cu130으로 돌아갑니다.

세션 한도보다 짧게 `TIME_BUDGET_MINUTES`를 잡으면 궤적 경계에서 깨끗하게 멈추므로,
반쯤 쓰인 파일이 남지 않습니다.

## 테스트

```bash
uv run pytest
```

`tests/test_week1.py`, `tests/test_week2.py`, `tests/test_week3.py`,
`tests/test_week4.py`, `tests/test_week5.py`, `tests/test_week6.py`는 네트워크 없이
실행됩니다. 6주차 테스트는 학습 자체를 교체 가능한 `trainer` 인자로 두어 GPU 없이도 선택
전략·예산 회계·재개 동작을 전부 검증합니다. 통합 테스트는 이미 내려받은 데이터를
재사용하며, 데이터가 없으면 최초 한 번 ESOL을 다운로드합니다.

재사용 가능한 1주차 API는 `src/ai4sci_molecule/week1.py`에 있습니다.

- `load_esol(root)`
- `smiles_to_graph(smiles, target=None)`
- `calculate_descriptors(smiles_list)`
- `make_random_split(n_samples, seed=42)`
- `build_baselines(seed=42)`
- `regression_metrics(y_true, y_pred)`

재사용 가능한 2주차 API는 `src/ai4sci_molecule/week2.py`에 있습니다.

- `GCNRegressor(...)`
- `GINRegressor(...)` (bond feature를 사용하는 GINE 구현)
- `TrainingConfig(...)`
- `build_gnn_models(...)`
- `train_gnn(model, dataset, split, ...)`
- `predict_gnn(result, dataset, indices, ...)`

재사용 가능한 3주차 API는 `src/ai4sci_molecule/week3.py`에 있습니다.

- `murcko_scaffold(smiles, ...)` / `scaffold_groups(smiles_list, ...)`
- `make_scaffold_split(smiles_list, seed=None, ...)` — `make_random_split`과 동일한
  `{"train", "validation", "test"}` 계약을 따릅니다
- `scaffold_statistics(...)` / `scaffold_table(...)` / `split_scaffold_overlap(...)`
- `nearest_neighbour_similarity(query_smiles, reference_smiles, ...)`
- `target_shift_table(targets, splits)`
- `SweepConfig(...)` / `run_generalization_sweep(dataset, ...)` / `load_or_run_sweep(...)`
- `aggregate_metrics(...)` / `generalization_gap(...)` /
  `similarity_error_correlation(...)` / `bin_similarity_errors(...)`
- `plot_*(...)` — 모두 `Figure`를 반환하므로 노트북에서 표시하고 `save_figures(...)`로
  같은 그림을 저장합니다

재사용 가능한 4주차 API는 `src/ai4sci_molecule/week4.py`에 있습니다. 학습을 하지 않으므로
`torch`를 import하지 않고, 3주차 예측 프레임만 입력으로 받습니다.

- `molecular_properties(smiles_list)` / `property_table(predictions)` — 1주차 descriptor
  5개를 재사용해 13개로 확장합니다
- `pool_split_types(predictions, ...)` / `effective_sample_sizes(predictions)` —
  `random → in_domain`, `scaffold`·`scaffold_shuffled` → `out_of_scaffold`
- `build_error_profile(predictions, ...)` — seed를 평균한 분자 단위 표.
  `residual = predicted − actual`이고, `Dummy (mean)`의 오차를 내재적 난이도
  (`baseline_absolute_error`)로 함께 실어 줍니다
- `rank_correlation(...)` / `partial_correlation(...)` /
  `bootstrap_correlation_interval(...)` — 퇴화 입력에서는 예외 대신 `nan`을 돌려줍니다
- `property_error_correlation(...)` — ρ + bootstrap CI + Dummy 통제군 + partial ρ +
  `trained_on` 순환성 플래그
- `bin_property_errors(profile, property_name, ...)` / `property_bin_summary(...)` —
  사분위 구간별 ratio-of-means 정규화
- `error_property_importance(...)` — held-out $R^2$를 `trustworthy` 플래그와 함께 먼저
  보고하고, permutation importance는 held-out fold에서 측정합니다
- `model_error_agreement(...)` / `model_error_disagreement(...)` — 같은 표현 family끼리
  같은 분자에서 틀리는지 확인합니다
- `shrinkage_fit(...)` / `scaffold_error_table(...)` / `case_studies(...)`
- `AnalysisConfig(...)` / `run_error_analysis(...)` / `load_or_run_error_analysis(...)` /
  `build_figures(...)` / `summarise(...)`
- `plot_*(...)` — 3주차와 같은 계약으로 `Figure`를 반환합니다

재사용 가능한 5주차 API는 `src/ai4sci_molecule/week5.py`에 있습니다. 새 모델을 만들지 않고
2주차의 `train_gnn` / `predict_gnn`을 그대로 감쌉니다.

- `UncertaintyConfig(...)` / `run_ensemble_sweep(dataset, ...)` / `build_ensembles(members, ...)`
  — `ensemble_size`는 **아키텍처당** 멤버 수이므로 `GIN+MLP`는 그 두 배를 모읍니다
- `gaussian_nll(...)` / `calibration_curve(...)` / `miscalibration_area(...)` /
  `coverage_at(...)` / `sharpness(...)` — 전부 순수 함수이고, 퇴화 입력에서는 예외 대신 `nan`
- `Calibrator` / `fit_calibrator(..., kind=...)` — `none`, `variance_scaling`(곱셈),
  `variance_offset`(빠진 aleatoric floor $\sqrt{\sigma^2+\tau^2}$), `conformal`(Gaussian 가정
  없이 validation 잔차 분위수). 정규분위수는 scipy 없이 `statistics.NormalDist`로 구합니다
- `risk_coverage_curve(...)` — model·oracle·random 세 곡선을 함께 냅니다. 두 경계 없이는 곡선을
  해석할 수 없습니다
- `separation_auroc(positive, negative)` — 순위로 계산한 Mann-Whitney AUC
- `collapse_seeds(...)` — bootstrap 전에 분자 단위로 접습니다. 접지 않으면 CI가 $\sqrt{3}$배
  좁아집니다
- `accuracy_table` / `ranking_table` / `risk_coverage_table` / `risk_coverage_summary_table` /
  `calibration_table` / `calibration_summary_table` / `calibration_novelty_table` /
  `separation_table` / `ensemble_size_table`
- `run_uncertainty_study(...)` / `analyse_sweep(sweep, ...)` /
  `load_or_run_uncertainty_study(...)` / `build_uncertainty_figures(...)` /
  `summarise_uncertainty(...)`
- `plot_*(...)` — 3·4주차와 같은 계약으로 `Figure`를 반환합니다

재사용 가능한 6주차 API는 `src/ai4sci_molecule/week6.py`에 있습니다. 새 모델을 만들지 않고
2주차의 `train_gnn` / `predict_gnn`을 5주차와 같은 방식으로 감쌉니다.

- `AcquisitionConfig(...)` — `budgets` / `regimes` / `trajectories` / `training_fingerprint` /
  `validate_pool(pool_size)`. fingerprint 설정도 학습 설정으로 셉니다(선택을 바꾸므로)
- `SelectionContext(...)` + `select_random` / `select_uncertainty` / `select_diversity` /
  `select_uncertainty_diversity` / `select_oracle_error` — 전부 순수 함수이고
  `ACQUISITION_FUNCTIONS` 레지스트리로 묶여 있습니다. `oracle_error`는 상한선일 뿐
  실행 가능한 전략이 아닙니다
- `similarity_matrix(query, reference, ...)` — 3주차 `nearest_neighbour_similarity`가 최대값만
  주는 데 반해 전체 Tanimoto 행렬을 줍니다. 배치 내 중복 제거에 필요합니다
- `experiment_pool(split)` / `initial_labelled_set(pool, seed=, size=)` /
  `internal_split(labelled, ...)` — test는 끝까지 획득 불가이고, early stopping validation도
  **측정된 분자이므로 예산에 포함**됩니다
- `train_ensemble(...)` / `EnsemblePrediction` — 학습은 교체 가능한 `trainer` 인자입니다.
  테스트는 여기에 값싼 결정적 대역을 끼워 GPU 없이 전체 루프를 검증합니다
- `run_active_learning_loop(...)` — 궤적 하나. `run_acquisition_sweep(..., resume=True,
  time_budget_minutes=...)` — **끊긴 자리에서 이어서 돌리는** sweep. 궤적이 끝난 뒤에만
  shard를 원자적으로 쓰고, 시간이 다 되면 궤적 경계에서 멈춥니다
- `learning_curve_table` / `budget_efficiency_table` / `strategy_comparison_table` /
  `batch_composition_table` / `scaffold_coverage_table` / `selection_profile_table` /
  `cold_start_table`
- `analyse_acquisition_sweep(...)` / `load_or_run_active_learning_study(...)` /
  `build_active_learning_figures(...)` / `summarise_active_learning(...)`
- `plot_*(...)` — 3·4·5주차와 같은 계약으로 `Figure`를 반환합니다

1·2주차는 고정 seed 42의 동일 index split(ESOL 기준 train 902 / validation 113 /
test 113)을 사용하고, validation RMSE로 early stopping checkpoint를 선택한 뒤 test를
한 번 평가합니다. 3주차는 같은 크기의 scaffold split을 추가로 만들어 두 regime을 seed
5개로 비교합니다. 4주차는 그 결과를 재사용해 분자 단위 오차를 property와 대조합니다. 5주차는
`random`과 `scaffold_shuffled` 두 regime에서 outer seed 3개씩, 각 split 위에 멤버 5개짜리
앙상블을 새로 학습합니다(3주차의 결정적 `scaffold` split은 outer seed가 무의미해지므로 쓰지
않습니다). 6주차는 그 두 regime에서 outer seed 5개씩, 각 split의 test를 끝까지 떼어 둔 채
labeled 100개로 시작해 라운드당 50개씩 8라운드를 사서 500개까지 갑니다. 라운드 0의 앙상블은
전략과 무관하게 같으므로 한 번만 학습해 공유합니다. 결과 해석은
[`reports/week3.md`](reports/week3.md), [`reports/week4.md`](reports/week4.md),
[`reports/week5.md`](reports/week5.md), [`reports/week6.md`](reports/week6.md)를 참고하세요.
