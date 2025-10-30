import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Any

class MemoryDataLoader:
    """메모리 기반 Word2Vec 데이터 로더"""

    NEGATIVE_TABLE_SIZE = int(1e8)

    def __init__(self,
                 sentences: List[List[int]],
                 word2id: Dict[str, int],
                 id2word: Dict[int, str],
                 word_frequency: Dict[int, int],
                 min_count: int = 5,
                 random_seed: int = 42):

        self.sentences = sentences
        self.word2id = word2id
        self.id2word = id2word
        self.word_frequency = word_frequency
        self.min_count = min_count
        self.random_seed = random_seed

        # NumPy random generator 생성 (재현성 보장)
        self.rng = np.random.default_rng(random_seed)
        
        # 통계 정보
        self.sentences_count = len(sentences)
        self.token_count = sum(len(sentence) for sentence in sentences)
        self.vocab_size = len(word2id)
        
        # Negative sampling과 subsampling 테이블 초기화
        self.negatives = []
        self.discards = []
        self.negpos = 0
        
        self._init_negative_table()
        self._init_discard_table()
        
        print(f"Loaded {self.sentences_count} sentences with {self.token_count} tokens")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _init_negative_table(self):
        """Negative sampling을 위한 테이블 초기화"""
        if not self.word_frequency:
            return

        # 0.75제곱으로 빈도 조정 (원래 Word2Vec 논문 설정)
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * self.NEGATIVE_TABLE_SIZE)

        # 각 단어 ID를 빈도에 비례해서 테이블에 추가
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)

        self.negatives = np.array(self.negatives)
        self.rng.shuffle(self.negatives)

        print(f"Negative sampling table created with {len(self.negatives)} entries")
    
    def _init_discard_table(self):
        """Subsampling을 위한 유지 확률(keep probability) 테이블 초기화"""
        if not self.word_frequency:
            return
            
        # Mikolov et al. subsampling: p_keep ≈ (sqrt(f/t) + 1) * (t/f)
        # f: 단어 상대빈도, t: 임계값(보통 1e-5~1e-3)
        t = 0.0001  # threshold for subsampling
        f = np.array(list(self.word_frequency.values())) / self.token_count
        keep_probs = (np.sqrt(f / t) + 1.0) * (t / f)
        self.discards = np.minimum(keep_probs, 1.0)
        
    def get_negatives(self, target: int, size: int) -> np.ndarray:
        """Negative samples 반환 (pos와 동일 ID 제외)"""
        if len(self.negatives) == 0:
            response = self.rng.integers(0, self.vocab_size, size)
        else:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                # 테이블 끝에 도달하면 처음부터 다시
                remaining = size - len(response)
                additional = self.negatives[:remaining]
                response = np.concatenate((response, additional))

        # 인덱스가 vocab_size를 초과하지 않도록 클리핑
        response = np.clip(response, 0, self.vocab_size - 1)

        # target과 동일한 음성 샘플은 재샘플링하여 치환
        if self.vocab_size > 1:
            mask = (response == target)
            if np.any(mask):
                # 동일한 개수만큼 재샘플 후 치환 (최소 충돌 회피)
                resample = self.rng.integers(0, self.vocab_size, mask.sum())
                # 혹시 또 target이 나올 수 있으니 한 번 더 보정
                second_mask = (resample == target)
                if np.any(second_mask):
                    resample[second_mask] = (resample[second_mask] + 1) % self.vocab_size
                response[mask] = resample

        return response
    
    def should_discard(self, word_id: int) -> bool:
        """Subsampling: 해당 단어를 버릴지 결정 (keep 확률 기반)"""
        if word_id >= len(self.discards) or word_id < 0:
            return False
        keep_prob = self.discards[word_id]
        return self.rng.random() > keep_prob


class MemoryWord2vecDataset(Dataset):
    """메모리 기반 Word2Vec 데이터셋"""
    
    def __init__(self, data_loader: MemoryDataLoader, window_size: int = 5):
        self.data_loader = data_loader
        self.window_size = window_size
        self.sentences = data_loader.sentences
        
        # 모든 훈련 페어를 미리 생성
        self.training_pairs = self._generate_all_pairs()
        
        print(f"Generated {len(self.training_pairs)} training pairs")
    
    def _generate_all_pairs(self) -> List[Tuple[int, int, np.ndarray]]:
        """모든 Skip-gram 훈련 페어를 미리 생성"""
        pairs = []
        
        for sentence in self.sentences:
            if len(sentence) <= 1:
                continue
            
            # 인덱스 유효성 검사
            valid_sentence = []
            for word_id in sentence:
                if 0 <= word_id < self.data_loader.vocab_size:
                    valid_sentence.append(word_id)
                else:
                    print(f"Warning: Invalid word_id {word_id}, vocab_size: {self.data_loader.vocab_size}")
            
            if len(valid_sentence) <= 1:
                continue
            
            # Subsampling: 빈도가 높은 단어들을 확률적으로 제거
            filtered_sentence = []
            for word_id in valid_sentence:
                if not self.data_loader.should_discard(word_id):
                    filtered_sentence.append(word_id)
            
            if len(filtered_sentence) <= 1:
                continue
            
            # Skip-gram 페어 생성
            for i, center_word in enumerate(filtered_sentence):
                # 윈도우 크기를 랜덤하게 조정 (1 ~ window_size)
                actual_window = self.data_loader.rng.integers(1, self.window_size + 1)

                # 문맥 단어들 추출
                start = max(0, i - actual_window)
                end = min(len(filtered_sentence), i + actual_window + 1)
                
                for j in range(start, end):
                    if i != j:  # 중심 단어 제외
                        context_word = filtered_sentence[j]
                        
                        # Negative samples 생성
                        neg_samples = self.data_loader.get_negatives(context_word, 5)
                        
                        pairs.append((center_word, context_word, neg_samples))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[int, int, np.ndarray]:
        """인덱스에 해당하는 훈련 페어 반환"""
        if idx >= len(self.training_pairs):
            idx = idx % len(self.training_pairs)
        
        return self.training_pairs[idx]
    
    @staticmethod
    def collate_fn(batch: List[Tuple[int, int, np.ndarray]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """배치 데이터를 텐서로 변환"""
        if not batch:
            return torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])
        
        center_words = [item[0] for item in batch]
        context_words = [item[1] for item in batch]
        
        # negative samples를 numpy array로 먼저 변환한 후 tensor로 변환
        negative_samples_list = [item[2] for item in batch]
        negative_samples_array = np.array(negative_samples_list)  # Warning 해결
        
        return (
            torch.LongTensor(center_words),
            torch.LongTensor(context_words), 
            torch.LongTensor(negative_samples_array)
        )


# 사용 예시
if __name__ == "__main__":
    # 예시 데이터
    sentences = [
        [0, 1, 2, 3],  # "king queen royal palace"의 인덱스
        [1, 4, 5],     # "queen woman person"의 인덱스
        [0, 6, 7]      # "king man strong"의 인덱스
    ]
    
    word2id = {"king": 0, "queen": 1, "royal": 2, "palace": 3, 
               "woman": 4, "person": 5, "man": 6, "strong": 7}
    id2word = {v: k for k, v in word2id.items()}
    word_frequency = {0: 10, 1: 8, 2: 5, 3: 3, 4: 6, 5: 4, 6: 7, 7: 2}
    
    # DataLoader 생성
    data_loader = MemoryDataLoader(
        sentences=sentences,
        word2id=word2id,
        id2word=id2word,
        word_frequency=word_frequency,
        min_count=1
    )
    
    # Dataset 생성
    dataset = MemoryWord2vecDataset(data_loader, window_size=2)
    
    # 첫 번째 훈련 페어 확인
    center, context, negatives = dataset[0]
    print(f"Center: {center} ({id2word[center]})")
    print(f"Context: {context} ({id2word[context]})")
    print(f"Negatives: {negatives}")
    
    # PyTorch DataLoader와 함께 사용
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # 배치 확인
    for batch_idx, (pos_u, pos_v, neg_v) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Center words: {pos_u}")
        print(f"  Context words: {pos_v}")
        print(f"  Negative samples shape: {neg_v.shape}")
        
        if batch_idx >= 2:  # 처음 몇 배치만 확인
            break