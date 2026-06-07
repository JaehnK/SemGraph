# SemGraph Refactoring Plan

작성일: 2026-05-24

이 문서는 기존 `REFACTORING_UML.md`와 `docs/SEMGRAPH_REFACTORING_AND_BRANCH_PLAN.md`를 통합한 단일 리팩토링 기준 문서다.
앞으로 Phase 번호는 이 문서의 번호만 사용한다.

## 1. 목표

1. `GRACE` 명칭과 실행 표면을 `SemGraph`로 통일한다.
2. 도메인 모델을 안정화하고 외부 라이브러리 의존을 줄인다.
3. 노드 feature 생성 경로를 BERT 중심으로 단순화한다.
4. spaCy, BERT, GraphMAE2, visualization, artifact 저장을 ports/adapters 경계로 이동한다.
5. Word2Vec, concat, attention fusion, 실험 코드는 main path에서 분리한다.
6. 저널 제출용 연구 플랫폼으로 재현성, 설치성, 실험 추적성을 개선한다.

## 2. 설계 원칙

- Domain first: entity는 외부 라이브러리 의존과 학습/전처리 실행 책임을 갖지 않는다.
- Application orchestration: pipeline은 use case 조합으로 표현한다.
- Ports and adapters: spaCy, transformers, DGL/GraphMAE2, matplotlib, JSON 저장은 adapter로 격리한다.
- Legacy isolation: Word2Vec, concat, attention fusion, 오래된 실험 경로는 `legacy/` 또는 `research/`로 분리한다.
- Compatibility first: 큰 이름 변경과 의미 변경을 한 커밋에 섞지 않는다.
- Reproducibility by construction: seed, 환경, artifact writer, 실험 설정을 명시적으로 관리한다.

## 3. 통합 Phase 기준

### Phase 0. Baseline And Runtime [부분 완료]

목표:

- 현재 테스트 상태와 import/runtime 의존성을 기록한다.
- heavy dependency, 데이터 파일 부재, Cython build 문제를 리팩토링 회귀와 분리한다.

검증:

- `git status --short`
- 합의된 smoke test 실행
- `rg "GRACE|grace"`와 `rg "Word2Vec|w2v|concat|attention|fusion"` 기준 출력 보관

상태:

- 부분 완료. `core/entities/README.md`에 entity Cython build와 Phase 2 domain smoke test 실행법을 기록했다.

### Phase 1. SemGraph Naming Transition [완료]

목표:

- `GRACE` 브랜드/클래스/출력명을 `SemGraph`로 전환한다.
- 기존 import는 전환 기간 동안 compatibility alias로 유지한다.

작업:

- `core/services/GRACE`에서 `core/services/semgraph`로 활성 경로를 이동한다.
- `GRACEConfig` -> `SemGraphConfig`
- `GRACEPipeline` -> `SemGraphPipeline`
- CLI banner, log, output prefix를 `SemGraph`/`semgraph` 기준으로 변경한다.

검증:

- `python -c "from core.services.semgraph import SemGraphConfig, SemGraphPipeline"`
- `python -c "from core.services.GRACE import GRACEConfig, GRACEPipeline"`
- `python pipelines/main.py --help`
- `rg "GRACE|grace" core pipelines tests` 결과가 compatibility/legacy 용도만 남는다.

상태:

- 완료. 브랜치 `refactor/semgraph-rename`에서 완료했고, 이후 `main`에 반영했다.

### Phase 2. Domain Stabilization [완료]

목표:

- 도메인 entity의 책임을 줄이고 안정적인 aggregate/invariant를 만든다.

작업:

- `Documents`를 `Corpus` 개념으로 재정의하고 기존 `Documents`는 compatibility alias로 유지한다.
- `Word`의 embedding 관련 상태를 entity에서 제거한다.
- `Sentence`는 처리 결과를 담는 경량 데이터 객체로 축소한다.
- spaCy/fallback 처리 로직은 sentence service 계층으로 이동한다.
- `WordGraph`는 입력 검증과 내부 상태 보호를 갖는 도메인 aggregate로 고정한다.
- PyG 변환은 호출 시점에만 의존성을 로드한다.

검증:

- `PYTHONPATH=core python -m pytest -q tests/entities/test_phase2_domain_stabilization.py`
- 변경 entity/service 파일 `py_compile`

상태:

- 완료. 브랜치 `refactor/phase-2-domain-stabilization`에서 완료했다.

### Phase 3. BERT-Only Feature Path [완료]

목표:

- main node feature 생성 경로를 BERT 기반 term embedding으로 단순화한다.

작업:

- config에서 `embedding_method` 선택지를 제거하거나 `bert`만 허용한다.
- `w2v_dim`, `fusion_type`, concat/attention 전용 validation을 제거한다.
- `embed_size`는 BERT 후처리 target dimension으로 재정의한다.
- `NodeFeatureHandler`에서 Word2Vec service 생성과 method 분기를 제거한다.
- `_get_w2v_embeddings()`, `_get_concat_embeddings()`, `_get_attention_embeddings()`를 제거한다.
- pipeline의 node feature 단계 log와 결과 config를 BERT-only 기준으로 정리한다.

검증:

- BERT feature tensor shape가 `[num_words, embed_size]`로 유지된다.
- main path import 시 `Word2VecService`가 생성되지 않는다.
- `rg "Word2Vec|w2v|fusion_type|embedding_method='concat'|embedding_method='attention'" core/services pipelines tests`가 legacy-only로 남는다.

권장 브랜치:

- `refactor/bert-only-node-features`

Phase 3 실행 체크리스트:

1. Config를 BERT-only 의미로 좁힌다.
   - 수정 파일:
     - `core/services/semgraph/SemGraphConfig.py`
     - compatibility가 필요하면 `core/services/GRACE/*Config*`
     - `pipelines/main.py`
     - 관련 config 생성 테스트
   - 커밋 시점:
     - config 생성과 validation이 BERT-only 기준으로 통과할 때 커밋한다.
   - 커밋 메시지:
     - `refactor: simplify config to bert node embeddings`
   - 커밋 전 검증:
     - `python -m py_compile core/services/semgraph/SemGraphConfig.py`
     - `python -c "from core.services.semgraph.SemGraphConfig import SemGraphConfig; print(SemGraphConfig.__name__)"`
   - 하지 않을 일:
     - `NodeFeatureHandler` 내부 분기 제거를 이 커밋에 섞지 않는다.
     - Word2Vec 파일을 삭제하지 않는다.

2. `NodeFeatureHandler`를 BERT-only provider로 축소한다.
   - 수정 파일:
     - `core/services/Graph/NodeFeatureHandler.py`
     - 필요 시 class docstring과 method 이름
   - 커밋 시점:
     - `calculate_embeddings()`가 method 인자 없이 BERT feature tensor를 만들 수 있을 때 커밋한다.
   - 커밋 메시지:
     - `refactor: remove node embedding method branching`
   - 커밋 전 검증:
     - `python -m py_compile core/services/Graph/NodeFeatureHandler.py`
     - `rg "Word2Vec|w2v|fusion_type|_get_concat|_get_attention" core/services/Graph/NodeFeatureHandler.py`
   - 하지 않을 일:
     - `core/services/Word2Vec/*`를 이동하거나 삭제하지 않는다.
     - `AttentionFusion.py`를 이동하거나 삭제하지 않는다.

3. Pipeline node feature 단계를 BERT-only 흐름으로 정리한다.
   - 수정 파일:
     - `core/services/semgraph/SemGraphPipeline.py`
     - compatibility wrapper가 있다면 `core/services/GRACE/*Pipeline*`
     - `pipelines/main.py`
   - 커밋 시점:
     - pipeline의 node feature log, 결과 config, 저장 metadata가 BERT-only 기준으로 정리됐을 때 커밋한다.
   - 커밋 메시지:
     - `refactor: simplify pipeline node feature step`
   - 커밋 전 검증:
     - `python -m py_compile core/services/semgraph/SemGraphPipeline.py pipelines/main.py`
     - 가능한 환경이면 `python pipelines/main.py --help`
   - 하지 않을 일:
     - GraphMAE 학습, clustering, metric 동작을 바꾸지 않는다.
     - output artifact schema를 불필요하게 바꾸지 않는다.

4. Ablation과 활성 테스트를 BERT-only 기준으로 갱신한다.
   - 수정 파일:
     - `core/services/Experiment/AblationService.py`
     - `pipelines/ablation_main.py`
     - `tests/*` 중 활성 SemGraph/config/ablation 테스트
   - 커밋 시점:
     - 제거된 `w2v`, `concat`, `attention` 선택지를 기대하는 테스트가 BERT-only 기대값으로 바뀌었을 때 커밋한다.
   - 커밋 메시지:
     - `test: update expectations for bert-only features`
   - 커밋 전 검증:
     - `python -m py_compile core/services/Experiment/AblationService.py pipelines/ablation_main.py`
     - 가능한 환경이면 `pytest tests/services/Experiment/test_ablation_service.py`
   - 하지 않을 일:
     - research 실험 재배치를 이 커밋에 섞지 않는다.

5. BERT-only smoke test를 추가한다.
   - 수정 파일:
     - `tests/*` 또는 `tests/services/Graph/*`
   - 커밋 시점:
     - main path에서 Word2Vec provider가 생성되지 않는다는 회귀 테스트가 통과할 때 커밋한다.
   - 커밋 메시지:
     - `test: cover bert-only node feature path`
   - 커밋 전 검증:
     - 새 smoke test 단독 실행
     - `rg "Word2Vec|w2v|concat|attention|fusion" core/services pipelines tests`
   - 하지 않을 일:
     - legacy 파일 삭제를 테스트 커밋에 섞지 않는다.

Phase 3 완료 조건:

- main path에서 `Word2VecService`가 생성되지 않는다.
- `NodeFeatureHandler`가 BERT feature 생성과 dimension adjustment만 담당한다.
- config와 pipeline에서 `w2v`, `concat`, `attention`, `fusion_type` 선택지가 사라지거나 legacy-only로 남는다.
- Word2Vec/AttentionFusion 파일은 아직 삭제하지 않고 Phase 6 대상으로 남긴다.
- 합의된 smoke test와 py_compile 검증이 통과한다.

상태:

- 완료. 브랜치 `refactor/bert-only-node-features`에서 main node feature path를 BERT-only 기준으로 단순화했다.
- 검증 기준은 `SemGraphConfig`, `NodeFeatureHandler`, `SemGraphPipeline`, ablation CLI py_compile, `pipelines/main.py --help`, `pipelines/ablation_main.py --help`, `PYTHONPATH=core` 기반 BERT-only smoke test로 둔다.
- `core/services/Word2Vec/*`, `core/services/Graph/AttentionFusion.py`, research/legacy 실험 재배치는 Phase 6에서 처리한다.

### Phase 4. Ports And Adapters [완료]

목표:

- 구현 클래스 중심 구조를 ports/adapters 구조로 전환한다.

작업:

- `CorpusRepository`, `TextPreprocessor`, `EmbeddingProvider`, `RepresentationLearner`, `Clusterer`, `ArtifactWriter` 인터페이스를 도입한다.
- spaCy, BERT, GraphMAE2, matplotlib, JSON 저장을 adapter로 이동한다.
- `GraphService`의 DGL 변환 책임을 adapter로 분리한다.
- `WordGraph`는 도메인 그래프 상태만 유지한다.

검증:

- domain/entity import가 spaCy, DGL, transformers, matplotlib 없이 가능하다.
- GraphMAE 학습과 clustering smoke test가 기존 결과 구조를 유지한다.

권장 브랜치:

- `refactor/phase-4-ports-adapters`

Phase 4 실행 체크리스트:

1. Application port 계약을 먼저 정의한다.
   - 수정 파일:
     - `core/services/ports/*`
     - port import smoke test
   - 커밋 시점:
     - port 모듈이 spaCy, transformers, DGL, matplotlib 없이 import될 때 커밋한다.
   - 커밋 메시지:
     - `refactor: define application service ports`
   - 하지 않을 일:
     - 기존 service 구현을 이 커밋에서 이동하지 않는다.

2. Artifact 저장을 `ArtifactWriter` adapter로 분리한다.
   - 수정 파일:
     - `core/services/adapters/*`
     - `core/services/semgraph/SemGraphPipeline.py`
     - 저장 관련 smoke test
   - 커밋 시점:
     - 결과 JSON 저장 경로와 schema가 유지될 때 커밋한다.
   - 커밋 메시지:
     - `refactor: route semgraph artifacts through writer adapter`
   - 하지 않을 일:
     - visualization 저장까지 함께 옮기지 않는다.

3. BERT node feature provider를 `EmbeddingProvider` adapter 경계로 감싼다.
   - 수정 파일:
     - `core/services/Graph/NodeFeatureHandler.py`
     - `core/services/DBert/*` 또는 adapter wrapper
   - 커밋 시점:
     - `NodeFeatureHandler`가 concrete `BertService` 생성 책임을 직접 갖지 않을 때 커밋한다.
   - 커밋 메시지:
     - `refactor: inject bert embedding provider`
   - 하지 않을 일:
     - BERT embedding 계산 결과 의미를 바꾸지 않는다.

4. GraphMAE/DGL 변환을 representation adapter 경계로 분리한다.
   - 수정 파일:
     - `core/services/GraphMAE/*`
     - `core/services/Graph/GraphService.py`
     - adapter smoke test
   - 커밋 시점:
     - GraphMAE 학습 호출 결과 구조가 유지되고, main pipeline import 시 GraphMAE2가 불필요하게 로드되지 않을 때 커밋한다.
   - 커밋 메시지:
     - `refactor: isolate graphmae representation adapter`
   - 하지 않을 일:
     - GraphMAE2 fork 내부 파일은 건드리지 않는다.

5. spaCy text preprocessing을 text preprocessor adapter 경계로 분리한다.
   - 수정 파일:
     - `core/services/Document/*`
     - adapter wrapper 또는 factory
     - import smoke test
   - 커밋 시점:
     - domain/entity import와 lightweight service import가 spaCy 없이 가능할 때 커밋한다.
   - 커밋 메시지:
     - `refactor: isolate spacy text preprocessing adapter`
   - 하지 않을 일:
     - sentence/domain entity 책임을 다시 늘리지 않는다.

Phase 4 완료 조건:

- application-facing ports가 존재하고 heavy implementation import와 분리된다.
- main pipeline의 artifact 저장은 writer adapter를 통해 수행된다.
- BERT, spaCy, GraphMAE2/DGL, matplotlib 의존성은 호출 시점 adapter 뒤로 이동한다.
- `core/GraphMAE2` nested repo는 이번 Phase에서도 직접 수정하지 않는다.

상태:

- 완료. 브랜치 `refactor/phase-4-ports-adapters`에서 완료했고, 이후 `main`에 반영했다.

### Phase 5. Main Path Legacy Cleanup [완료]

목표:

- 모델 설계가 확정된 상태에서 main 실행 경로에 남아 있는 legacy embedding 의존성을 제거한다.
- Word2Vec, concat, attention fusion은 SemGraph의 현재 방법론이 아니므로 main path와 분리한다.

작업:

- `core/services/Word2Vec/*`, `core/entities/skipgram.py`, `core/services/Graph/AttentionFusion.py`의 main path 참조를 확인한다.
- main path에서 더 이상 참조되지 않는 legacy embedding 코드는 `legacy/` 또는 `research/`로 격리하거나 삭제 후보로 표시한다.
- `core/services/__init__.py`의 Word2Vec lazy export를 제거하거나 legacy-only로 명시한다.
- pipeline/config/ablation import에서 Word2Vec, concat, attention fusion 잔여 참조를 제거한다.
- GraphMAE2 내부 GAT attention, 일반 `torch.cat`/`np.concatenate` 같은 구현 용어는 제거 대상에서 제외한다.

검증:

- `rg "Word2Vec|w2v|fusion_type|AttentionFusion|embedding_method|concat" core/services pipelines core/entities`
- main pipeline import 시 Word2Vec, skipgram, AttentionFusion이 로드되지 않는다.
- `python -c "from core.services.semgraph import SemGraphConfig, SemGraphPipeline"`
- `python pipelines/main.py --help`
- 관련 파일 `py_compile`

상태:

- 완료. 브랜치 `refactor/legacy-embedding-cleanup`에서 Word2Vec, skipgram, AttentionFusion, GRACE compatibility wrapper를 제거했다.
- `embedding_method`는 BERT-only 설정 검증과 ablation 결과 키로만 유지한다.

완료 조건:

- SemGraph main path는 BERT-only node feature, GraphMAE representation, clustering 흐름만 가진다.
- legacy embedding 경로는 main path에서 분리되어 후속 실험 설계를 오염시키지 않는다.
- 삭제 여부가 애매한 코드는 즉시 삭제보다 legacy 격리를 우선한다.

권장 브랜치:

- `refactor/legacy-embedding-cleanup`

### Phase 6. Environment And Reproducibility [예정]

목표:

- ablation과 신규 데이터셋 실험 전에 재현 가능한 실행 환경을 확정한다.
- 기존 conda freeze 성격의 의존성에서 벗어나 `uv` 기반 환경으로 정리한다.

작업:

- `pyproject.toml`을 도입해 runtime/dev dependency를 구분한다.
- `uv.lock`을 생성해 설치 상태를 고정한다.
- 기존 `requirements.txt`는 legacy snapshot으로 보존하거나 ignore 유지한다.
- Cython build, spaCy model, torch/PyG/DGL/GraphMAE2 설치 흐름을 명시한다.
- `uv run` 기준 CLI, import smoke, test 실행법을 정리한다.
- seed, config, artifact writer, run metadata 관리 방식을 표준화한다.

검증:

- `uv sync`
- `uv run python -c "from core.services.semgraph import SemGraphConfig, SemGraphPipeline"`
- `uv run python pipelines/main.py --help`
- 합의된 smoke test set 통과
- GPU 검증은 CPU/import smoke와 분리해서 기록한다.

권장 브랜치:

- `chore/uv-environment`

### Phase 7. Experiment Protocol Build [예정]

목표:

- 데이터셋 추가, ablation 설계, 평가 지표, 최종 실험 프로토콜을 하나의 단계로 확정한다.
- SemGraph가 일반 텍스트뿐 아니라 학술 텍스트/science mapping 문제에서도 타당한지 검증할 수 있게 한다.

작업:

- 기존 AG News는 proof-of-concept/연결 실험으로 유지한다.
- arXiv, OpenAlex, PubMed 중 1~2개 학술 데이터셋을 추가한다.
- dataset config에 `csv_path`, `text_column`, `label_column`, metadata, output path를 명확히 둔다.
- 외부 라벨 평가가 가능하도록 arXiv category, OpenAlex concept, PubMed MeSH 등 metadata를 보존한다.
- ablation은 BERT-only SemGraph 기준으로 재정의한다.
  - mask rate
  - epochs
  - output/hidden dimension
  - encoder/decoder type
  - edge weight threshold
  - edge top-k
  - cluster k selection
- 평가 지표를 확정한다.
  - NPMI
  - Silhouette, Davies-Bouldin, Calinski-Harabasz
  - Modularity
  - external label alignment
  - temporal or seed stability
- 논문용 결과 저장 구조와 표 생성 방식을 정리한다.

검증:

- 신규 데이터셋 준비 스크립트 smoke 실행
- dataset별 `pipelines/main.py --config ... --max-docs ...` smoke 실행
- ablation runner가 dataset/config별 결과 JSON을 생성한다.
- 반복 seed 결과가 평균/표준편차로 집계된다.

권장 브랜치:

- `feature/experiment-protocol`
- 필요 시 하위 브랜치:
  - `feature/add-scholar-datasets`
  - `refactor/bert-only-ablation-protocol`
  - `feature/evaluation-metrics`

## 4. 브랜치 진행 기준

완료된 브랜치:

- `refactor/semgraph-rename`: Phase 1 기준
- `refactor/phase-2-domain-stabilization`: Phase 2 기준
- `refactor/bert-only-node-features`: Phase 3 기준
- `refactor/phase-4-ports-adapters`: Phase 4 기준

권장 다음 브랜치:

- `refactor/legacy-embedding-cleanup`: Phase 5 기준

이후 후보:

- `chore/uv-environment`
- `feature/experiment-protocol`
- `feature/add-scholar-datasets`
- `refactor/bert-only-ablation-protocol`
- `feature/evaluation-metrics`

## 5. 커밋 원칙

- phase 하나를 한 커밋으로 처리하지 않는다.
- 파일 이동, 이름 변경, 의미 변경, 테스트 변경을 가능한 한 분리한다.
- import가 깨진 중간 상태는 커밋하지 않는다.
- 기존 dirty state나 nested repo 변경은 해당 작업과 직접 관련이 없으면 포함하지 않는다.
- 삭제는 main path 참조가 끊긴 뒤 별도 커밋으로 처리한다.
- 커밋은 "수정 파일 범위가 설명 가능하고, 해당 범위의 검증 명령이 통과한 시점"에 만든다.
- 커밋 전에는 `git diff --stat`과 staged diff를 확인한다.

## 6. 현재 주의 사항

- `core/GraphMAE2`는 nested git repo 형태이며, 상위 repo의 Phase 작업과 별개로 관리한다.
- 모델 설계 방향은 SemGraph + BERT-only node feature + GraphMAE representation + clustering으로 확정된 상태다.
- Word2Vec, skipgram, attention fusion, GRACE compatibility wrapper는 Phase 5에서 main path에서 제거했다.
- 현재 로컬 환경은 `torch_geometric`이 없어 `from core.services import GraphService` 직접 import가 실패한다. Phase 6 uv 환경 정리에서 고정한다.
- 데이터셋 추가와 ablation은 분리된 작업이 아니라 Phase 7 `Experiment Protocol Build`에서 함께 다룬다.
- `docs/*`와 기존 연구 산출물은 과거 맥락을 담고 있을 수 있다. 앞으로의 기준은 이 문서만 사용한다.
