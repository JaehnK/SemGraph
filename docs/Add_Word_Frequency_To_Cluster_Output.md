# 클러스터 결과에 단어 빈도 추가 계획서

## 1. 현황 분석

### 1.1 현재 출력 형식
현재 `grace_results_YYYYMMDD_HHMMSS.json` 파일의 클러스터 출력 형식:
```json
{
  "clusters": {
    "0": ["say", "company", "report", "monday", "tuesday", "friday", "price", "high"],
    "1": ["government", "official", "minister", "president"],
    ...
  }
}
```

### 1.2 목표 출력 형식
단어별 빈도와 클러스터별 예시 문장을 포함한 새로운 형식:
```json
{
  "clusters": {
    "0": {
      "words": [
        {"word": "say", "frequency": 1523},
        {"word": "company", "frequency": 1421},
        {"word": "report", "frequency": 1203},
        {"word": "monday", "frequency": 856}
      ],
      "sample_documents": [
        "The company said it would release earnings report next monday.",
        "Officials say the company reported strong results on monday.",
        ...
      ]
    },
    "1": {
      "words": [
        {"word": "government", "frequency": 2341},
        {"word": "official", "frequency": 1876},
        {"word": "minister", "frequency": 1654}
      ],
      "sample_documents": [
        "The government and officials including the minister announced new policy.",
        "Government minister and official representatives met today.",
        ...
      ]
    }
  }
}
```

**설명**:
- 각 클러스터는 `words` 리스트와 `sample_documents` 리스트를 포함
- `sample_documents`는 해당 클러스터의 단어들을 **가장 많이 포함**한 문장들 (상위 10개)
- 예: 클러스터 0의 4개 단어 중 3개 이상을 포함한 문장들을 우선적으로 선택

## 2. 관련 코드 구조

### 2.1 Word 클래스 (`core/entities/word.py`)
- `Word` 객체는 `freq` 속성에 단어 빈도를 저장
- `WordGraph.words[i].freq`로 접근 가능

### 2.2 WordTrie 클래스 (`core/entities/trie.pyx`)
- Trie 구조에서 각 노드는 `Word` 객체를 참조
- `Word` 객체의 `freq` 속성이 문서 내 출현 빈도를 저장
- `insert_or_get_word()` 메서드에서 `increment_freq()`를 통해 빈도 증가

### 2.3 GRACEPipeline 클래스 (`core/services/GRACE/GRACEPipeline.py`)
**Line 416-423**: `_build_cluster_info()` 메서드
```python
def _build_cluster_info(self) -> Dict[int, List[str]]:
    """클러스터별 단어 정보 구성"""
    cluster_info = {}
    for cluster_id in np.unique(self.cluster_labels):
        indices = np.where(self.cluster_labels == cluster_id)[0]
        words = [self.word_graph.words[i].content for i in indices]
        cluster_info[int(cluster_id)] = words
    return cluster_info
```

**Line 69, 161-162**: 문서와 문장 데이터 저장
```python
self.documents: Optional[List[str]] = None  # 원본 문서 저장
self.doc_service.create_sentence_list(documents=self.documents)
# doc_service.sentence_list에 모든 Sentence 객체 저장
```

### 2.4 Sentence 클래스 (`core/entities/sentence.py`)
- `Sentence.raw`: 원본 문장 텍스트
- `Sentence.word_objects`: 문장에 포함된 Word 객체 리스트
- 각 문장은 어떤 단어들을 포함하는지 알 수 있음

## 3. 구현 계획

### 3.1 핵심 수정사항

#### Step 1: 클러스터 단어들을 가장 많이 포함한 문장 찾기 메서드 추가
**파일**: `core/services/GRACE/GRACEPipeline.py`

새로운 헬퍼 메서드 추가:
```python
def _get_sample_sentences_for_cluster(self, cluster_words: List[str], max_samples: int = 10) -> List[str]:
    """
    클러스터의 단어들을 가장 많이 포함한 문장들을 반환

    Args:
        cluster_words: 클러스터에 속한 단어 리스트
        max_samples: 반환할 최대 문장 수

    Returns:
        클러스터 단어를 많이 포함한 문장 리스트 (매칭 개수 내림차순 정렬)
    """
    if self.doc_service is None or self.doc_service.sentence_list is None:
        return []

    # 각 문장에 대해 클러스터 단어 매칭 개수 계산
    sentence_scores = []
    cluster_words_lower = set(w.lower() for w in cluster_words)

    for sentence in self.doc_service.sentence_list:
        # 문장에 포함된 단어들 중 클러스터 단어와 매칭되는 개수 계산
        sentence_words = set(w.content.lower() for w in sentence.word_objects)
        match_count = len(sentence_words & cluster_words_lower)

        if match_count > 0:  # 최소 1개 이상 매칭된 문장만
            sentence_scores.append((sentence.raw, match_count))

    # 매칭 개수 기준 내림차순 정렬
    sentence_scores.sort(key=lambda x: x[1], reverse=True)

    # 상위 max_samples개 문장만 반환
    return [sent for sent, _ in sentence_scores[:max_samples]]
```

#### Step 2: `_build_cluster_info()` 메서드 수정
**파일**: `core/services/GRACE/GRACEPipeline.py` (Line 416-423)

**수정 전**:
```python
def _build_cluster_info(self) -> Dict[int, List[str]]:
    """클러스터별 단어 정보 구성"""
    cluster_info = {}
    for cluster_id in np.unique(self.cluster_labels):
        indices = np.where(self.cluster_labels == cluster_id)[0]
        words = [self.word_graph.words[i].content for i in indices]
        cluster_info[int(cluster_id)] = words
    return cluster_info
```

**수정 후**:
```python
def _build_cluster_info(self, include_samples: bool = True, max_samples: int = 10) -> Dict[int, Dict[str, Any]]:
    """
    클러스터별 단어 정보 구성 (빈도 및 예시 문장 포함)

    Args:
        include_samples: 예시 문장 포함 여부
        max_samples: 클러스터당 최대 예시 문장 수

    Returns:
        클러스터 정보 딕셔너리
        {
            cluster_id: {
                "words": [{"word": str, "frequency": int}, ...],
                "sample_documents": [str, ...]  # 클러스터 단어를 많이 포함한 문장들
            }
        }
    """
    cluster_info = {}

    for cluster_id in np.unique(self.cluster_labels):
        indices = np.where(self.cluster_labels == cluster_id)[0]

        # 단어 정보 수집 (빈도 포함)
        words_with_freq = [
            {
                "word": self.word_graph.words[i].content,
                "frequency": self.word_graph.words[i].freq
            }
            for i in indices
        ]

        # 빈도 기준 내림차순 정렬
        words_with_freq.sort(key=lambda x: x["frequency"], reverse=True)

        cluster_data = {
            "words": words_with_freq
        }

        # 예시 문장 추가
        if include_samples:
            cluster_words = [w["word"] for w in words_with_freq]
            cluster_data["sample_documents"] = self._get_sample_sentences_for_cluster(
                cluster_words,
                max_samples=max_samples
            )

        cluster_info[int(cluster_id)] = cluster_data

    return cluster_info
```

#### Step 3: 타입 힌트 업데이트
**위치**: 같은 파일의 관련 메서드들

영향받는 메서드:
- `_build_cluster_info()`: 반환 타입 변경
- `run()`: results 딕셔너리의 'clusters' 키 타입 변경

### 3.2 호환성 고려사항

#### 3.2.1 기존 시각화 코드 영향 확인
**파일**: `core/services/Visualization/VisualizationService.py`

`visualize_cluster_words()` 메서드가 단어 리스트를 사용하는지 확인 필요:
- **현재**: `cluster_words: Dict[int, List[str]]` 형식을 받는 것으로 추정
- **수정 필요**: `Dict[int, Dict[str, Any]]` 형식도 처리할 수 있도록 수정

```python
def visualize_cluster_words(self, cluster_words, filename, max_words=50):
    # 각 클러스터별로 처리
    for cluster_id, cluster_data in cluster_words.items():
        # 새로운 형식 처리
        if isinstance(cluster_data, dict) and "words" in cluster_data:
            word_list = [item["word"] for item in cluster_data["words"]]
        elif isinstance(cluster_data, list):
            # 이전 호환성 유지
            if len(cluster_data) > 0 and isinstance(cluster_data[0], dict):
                word_list = [item["word"] for item in cluster_data]
            else:
                word_list = cluster_data
        else:
            word_list = cluster_data
        # ... 워드클라우드 생성
```

#### 3.2.2 main.py 출력 로직 수정
**파일**: `main.py` (Line 137-143)

**수정 전**:
```python
if 'cluster_top_words' in results:
    print(f"\n{Fore.YELLOW}Top Words per Cluster:{Style.RESET_ALL}")
    for cluster_id, top_words in list(results['cluster_top_words'].items())[:5]:
        words = ', '.join(top_words[:10])
        print(f"\n  {Fore.CYAN}Cluster {cluster_id}{Style.RESET_ALL}:")
        print(f"    {words}")
```

**수정 후**:
```python
if 'clusters' in results:
    print(f"\n{Fore.YELLOW}Top Words per Cluster (with frequencies):{Style.RESET_ALL}")
    for cluster_id, cluster_data in list(results['clusters'].items())[:5]:
        # 상위 10개 단어만 표시
        word_items = cluster_data.get("words", [])[:10]
        words_display = ', '.join([f"{item['word']}({item['frequency']})" for item in word_items])
        print(f"\n  {Fore.CYAN}Cluster {cluster_id}{Style.RESET_ALL}:")
        print(f"    Words: {words_display}")

        # 예시 문장도 출력 (선택적)
        if 'sample_documents' in cluster_data and cluster_data['sample_documents']:
            print(f"    Sample: {cluster_data['sample_documents'][0][:100]}...")  # 첫 문장 100자만
```

### 3.3 추가 개선사항 (선택적)

#### Option 1: 빈도 통계 추가
각 클러스터에 대한 빈도 통계를 추가:
```python
cluster_data = {
    "words": words_with_freq,
    "sample_documents": sample_docs,
    "statistics": {
        "total_frequency": sum(w["frequency"] for w in words_with_freq),
        "avg_frequency": np.mean([w["frequency"] for w in words_with_freq]),
        "word_count": len(words_with_freq)
    }
}
```

#### Option 2: 매칭 개수 정보 추가
예시 문장에 몇 개의 클러스터 단어가 포함되었는지 정보 추가:
```python
def _get_sample_sentences_for_cluster(self, cluster_words: List[str], max_samples: int = 10) -> List[Dict[str, Any]]:
    # ... (기존 로직)
    # 반환 형식 변경:
    return [
        {
            "sentence": sent,
            "match_count": count,
            "match_ratio": count / len(cluster_words)
        }
        for sent, count in sentence_scores[:max_samples]
    ]
```

#### Option 3: 정규화된 빈도 추가
전체 문서 대비 상대 빈도 추가:
```python
total_freq = sum(w.freq for w in self.word_graph.words)
words_with_freq = [
    {
        "word": self.word_graph.words[i].content,
        "frequency": self.word_graph.words[i].freq,
        "relative_frequency": self.word_graph.words[i].freq / total_freq
    }
    for i in indices
]
```

## 4. 테스트 계획

### 4.1 단위 테스트
1. `_build_cluster_info()` 메서드 테스트
   - 빈도 정보가 올바르게 포함되는지 확인
   - 정렬이 올바르게 되는지 확인

### 4.2 통합 테스트
1. 전체 파이프라인 실행 테스트
   - JSON 출력 형식 확인
   - 시각화 호환성 확인
   - main.py 출력 확인

### 4.3 검증 항목
- [ ] JSON 파일에 빈도 정보 포함
- [ ] 빈도 기준 내림차순 정렬
- [ ] 시각화 기능 정상 동작
- [ ] main.py 콘솔 출력 정상 동작
- [ ] 기존 결과 파일과 호환성 유지 (읽기)

## 5. 구현 순서

1. **Phase 1**: 핵심 기능 구현
   - [ ] `GRACEPipeline._build_cluster_info()` 메서드 수정
   - [ ] 타입 힌트 업데이트

2. **Phase 2**: 호환성 수정
   - [ ] `VisualizationService.visualize_cluster_words()` 수정
   - [ ] `main.py` 출력 로직 수정

3. **Phase 3**: 테스트
   - [ ] 단위 테스트 작성 및 실행
   - [ ] 통합 테스트 실행
   - [ ] 결과 검증

4. **Phase 4**: 문서화 (선택적)
   - [ ] README 업데이트
   - [ ] 변경사항 문서화

## 6. 예상 영향 범위

### 6.1 수정 필요 파일
1. `core/services/GRACE/GRACEPipeline.py` - 핵심 수정
2. `core/services/Visualization/VisualizationService.py` - 호환성
3. `main.py` - 출력 형식

### 6.2 영향받지 않는 부분
- Word 클래스 (이미 freq 속성 존재)
- WordTrie 클래스 (빈도 관리 이미 구현)
- WordGraph 클래스 (단어 접근만 사용)
- 클러스터링 알고리즘
- GraphMAE 모델

## 7. 리스크 및 대응

### 7.1 리스크
1. **시각화 호환성 문제**
   - 대응: 이전/이후 형식 모두 처리 가능하도록 구현

2. **JSON 크기 증가**
   - 대응: 빈도 정보는 정수이므로 크기 증가 미미

3. **기존 분석 스크립트 영향**
   - 대응: 구 형식도 읽을 수 있는 헬퍼 함수 제공

### 7.2 롤백 계획
- Git을 통한 버전 관리
- 수정 전 브랜치 생성 권장
- 테스트 실패 시 즉시 rollback

## 8. 향후 확장 가능성

1. **단어별 메타데이터 추가**
   - POS 태그 분포
   - TF-IDF 점수
   - 문서 내 위치 정보

2. **클러스터 품질 지표**
   - 클러스터 내 단어 다양성
   - 평균 공출현 빈도
   - 클러스터 응집도

3. **대화형 결과 탐색**
   - Web UI로 빈도 기반 필터링
   - 단어 클릭 시 상세 정보 표시

---

**작성일**: 2025-10-30
**작성자**: Claude Code
**상태**: 계획 수립 완료
