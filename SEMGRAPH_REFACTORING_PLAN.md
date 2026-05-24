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

### Phase 0. Baseline And Runtime

목표:

- 현재 테스트 상태와 import/runtime 의존성을 기록한다.
- heavy dependency, 데이터 파일 부재, Cython build 문제를 리팩토링 회귀와 분리한다.

검증:

- `git status --short`
- 합의된 smoke test 실행
- `rg "GRACE|grace"`와 `rg "Word2Vec|w2v|concat|attention|fusion"` 기준 출력 보관

상태:

- 부분 완료. `core/entities/README.md`에 entity Cython build와 Phase 2 domain smoke test 실행법을 기록했다.

### Phase 1. SemGraph Naming Transition

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

- 이전 브랜치 `refactor/semgraph-rename`에서 진행된 것으로 본다.

### Phase 2. Domain Stabilization

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

### Phase 3. BERT-Only Feature Path

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

### Phase 4. Ports And Adapters

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

### Phase 5. Application Pipeline Simplification

목표:

- `SemGraphPipeline`의 과도한 orchestration 책임을 use case 단위로 줄인다.

작업:

- load/preprocess, graph build, node feature, GraphMAE train, clustering, evaluation/save를 use case 단위로 분리한다.
- artifact 저장/시각화/reporting을 pipeline 핵심 흐름에서 분리한다.
- 결과 파일명과 output path는 `semgraph_*` 기준으로 유지한다.

검증:

- `SemGraphPipeline(config).run()` smoke test가 성공한다.
- 결과 JSON과 embedding 저장 구조가 필요한 분석 도구와 호환된다.

### Phase 6. Research And Legacy Isolation

목표:

- 실험용 코드와 legacy embedding 경로를 production core에서 분리한다.

작업:

- Word2Vec, concat, attention fusion은 main path 참조를 끊은 뒤 `legacy/` 또는 `research/`로 이동한다.
- `experiments/`, `analysis/`, `viz/`는 `research/` 영역으로 재배치한다.
- journal용 experiment runner는 application/use case만 호출하도록 제한한다.
- ablation은 BERT-only 연구 질문에 맞게 재정의한다.

검증:

- main pipeline import 시 Word2Vec, skipgram, AttentionFusion이 로드되지 않는다.
- ablation unit test와 JSON serialization test가 통과한다.

권장 브랜치:

- `refactor/semgraph-ablation`
- `refactor/legacy-embedding-cleanup`

### Phase 7. Environment And Reproducibility

목표:

- 실행/테스트 환경과 실험 재현성을 명확히 한다.

작업:

- dependency metadata와 lockfile을 정리한다.
- Cython build, spaCy model, DGL/GraphMAE2 dependency 설치 경로를 명시한다.
- seed, config, artifact writer, run metadata를 표준화한다.

검증:

- 합의된 환경에서 `pytest` 또는 smoke test set 통과
- CLI help와 import smoke test 통과

권장 브랜치:

- `chore/uv-environment`

## 4. 브랜치 진행 기준

완료된 브랜치:

- `refactor/semgraph-rename`: Phase 1 기준
- `refactor/phase-2-domain-stabilization`: Phase 2 기준

권장 다음 브랜치:

- `refactor/bert-only-node-features`: Phase 3 기준

이후 후보:

- `refactor/ports-and-adapters`
- `refactor/semgraph-pipeline-usecases`
- `refactor/semgraph-ablation`
- `refactor/legacy-embedding-cleanup`
- `chore/uv-environment`

## 5. 커밋 원칙

- phase 하나를 한 커밋으로 처리하지 않는다.
- 파일 이동, 이름 변경, 의미 변경, 테스트 변경을 가능한 한 분리한다.
- import가 깨진 중간 상태는 커밋하지 않는다.
- 기존 dirty state나 nested repo 변경은 해당 작업과 직접 관련이 없으면 포함하지 않는다.
- 삭제는 main path 참조가 끊긴 뒤 별도 커밋으로 처리한다.

## 6. 현재 주의 사항

- `core/GraphMAE2`는 nested git repo 형태이며, 상위 repo의 Phase 작업과 별개로 관리한다.
- Word2Vec 경로는 아직 살아 있다. Phase 2에서는 `Word` entity에서 embedding 상태만 분리했고, W2V 서비스 제거는 Phase 3 이후 main path 참조가 끊긴 뒤 Phase 6에서 처리한다.
- `docs/*`와 기존 연구 산출물은 과거 맥락을 담고 있을 수 있다. 앞으로의 기준은 이 문서만 사용한다.
