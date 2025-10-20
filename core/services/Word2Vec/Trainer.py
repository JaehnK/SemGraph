import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional
import os
import numpy as np
from datetime import datetime
import random

class Word2VecTrainer:
    """메모리 기반 Word2Vec 훈련기"""

    def __init__(self,
                iterations: int = 10,
                initial_lr: float = 0.001,
                batch_size: int = 32,
                use_cuda: Optional[bool] = None,
                random_seed: int = 42):

        # 재현성을 위한 랜덤 시드 고정
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.iterations = iterations
        self.initial_lr = initial_lr
        self.batch_size = batch_size
        self.random_seed = random_seed

        # GPU 사용 설정
        self.use_cuda = use_cuda if use_cuda is not None else torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        # 학습률 스케줄 상태
        self.global_step = 0
        self.total_steps = 0
        self.min_lr_ratio = 1e-3  # 최종 lr = initial_lr * 1e-3

        print(f"Using device: {self.device}")
        print(f"Random seed: {random_seed} (for reproducibility)")
    
    def train(self, 
            model,  # SkipGramModel
            dataset,  # MemoryWord2vecDataset
            data_loader,  # MemoryDataLoader  
            output_file: str = "word2vec_embeddings.txt"):
        """모델 훈련 실행"""
        
        # 모델을 디바이스로 이동
        model.to(self.device)
        model.train()
        
        # DataLoader 생성
        torch_dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,  # 데이터 로딩 병렬화
            pin_memory=True,  # GPU 전송 속도 향상
            collate_fn=dataset.collate_fn
        )
        
        print(f"Starting training with {len(dataset)} training pairs")
        print(f"Batch size: {self.batch_size}, Iterations: {self.iterations}")
        
        # 전체 스텝 계산 및 옵티마이저 1회 초기화
        batches_per_epoch = max(1, len(torch_dataloader))
        self.total_steps = self.iterations * batches_per_epoch
        self.global_step = 0
        
        optimizer = optim.SGD(model.parameters(), lr=self.initial_lr, momentum=0.9)
        print(f"Optimizer: SGD(momentum=0.9), initial_lr={self.initial_lr}, final_lr={self.initial_lr * self.min_lr_ratio}")
        
        # 각 iteration별 훈련
        for iteration in range(self.iterations):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration + 1}/{self.iterations}")
            print(f"{'='*50}")
            
            self._train_epoch(model, torch_dataloader, iteration, optimizer)
            
        print(f"\nTraining completed!")
        
        return model
    
    def _train_epoch(self, model, dataloader, iteration: int, optimizer):
        """한 epoch 훈련"""
        
        running_loss = 0.0
        total_batches = len(dataloader)
        
        # 진행 상황 표시를 위한 tqdm
        progress_bar = tqdm(
            enumerate(dataloader), 
            total=total_batches,
            desc=f"Epoch {iteration + 1}",
            ncols=100,
            leave=False # 에포크 완료 후 프로그레스 바를 터미널에 남기지 않음
        )
        
        for batch_idx, sample_batched in progress_bar:
            # 배치가 비어있는지 확인
            if len(sample_batched[0]) == 0:
                continue
            
            # 데이터를 디바이스로 이동
            pos_u = sample_batched[0].to(self.device)
            pos_v = sample_batched[1].to(self.device)
            neg_v = sample_batched[2].to(self.device)
            
            # 선형 학습률 감소 적용
            current_lr = self._compute_linear_lr()
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # 순전파
            optimizer.zero_grad()
            loss = model.forward(pos_u, pos_v, neg_v)
            
            # 역전파
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 글로벌 스텝 증가
            self.global_step += 1

            # 손실 추적 (지수 이동 평균)
            running_loss = running_loss * 0.9 + loss.item() * 0.1
            
            # 진행 상황 업데이트
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f'{running_loss:.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            # 주기적으로 손실 출력 (tqdm postfix로 대체되므로 주석 처리)
            # if batch_idx > 0 and batch_idx % 500 == 0:
            #     print(f"\nBatch {batch_idx}/{total_batches} - Loss: {running_loss:.4f}")
        print(f"Epoch {iteration + 1} completed - Final loss: {running_loss:.4f}")

    def _compute_linear_lr(self) -> float:
        """초기 lr에서 최종 lr로 선형 감소한 현재 lr을 계산"""
        if self.total_steps <= 1:
            return self.initial_lr
        progress = min(1.0, max(0.0, self.global_step / (self.total_steps - 1)))
        start_lr = self.initial_lr
        end_lr = self.initial_lr * self.min_lr_ratio
        return start_lr + (end_lr - start_lr) * progress
    
    def evaluate_similarity(self, 
                            model, 
                            data_loader, 
                            test_words: list = None,
                            top_k: int = 5):
        """훈련된 모델의 단어 유사도 평가"""
        
        if test_words is None:
            # 빈도 높은 단어들로 기본 테스트
            word_freq_pairs = [(data_loader.id2word[wid], freq) 
                                for wid, freq in data_loader.word_frequency.items()]
            word_freq_pairs.sort(key=lambda x: x[1], reverse=True)
            test_words = [word for word, _ in word_freq_pairs[:10]]
        
        model.eval()
        print(f"\n{'='*50}")
        print("Word Similarity Evaluation")
        print(f"{'='*50}")
        
        for word in test_words:
            if word in data_loader.word2id:
                similar_words = self._find_similar_words(
                    model, data_loader, word, top_k
                )
                print(f"\n'{word}' 와 유사한 단어들:")
                for similar_word, similarity in similar_words:
                    print(f"  {similar_word}: {similarity:.4f}")
            else:
                print(f"'{word}' not found in vocabulary")
    
    def get_word_vector(self, model, data_loader, word: str) -> Optional[np.ndarray]:
        """단어의 numpy 벡터 반환"""
        import numpy as np
        
        if word not in data_loader.word2id:
            print(f"Warning: '{word}' not found in vocabulary")
            return None
        
        word_id = data_loader.word2id[word]
        
        # 모델을 평가 모드로 설정
        model.eval()
        
        with torch.no_grad():
            # u_embeddings에서 해당 단어의 벡터 추출
            word_vector = model.u_embeddings.weight[word_id].cpu().numpy()
        
        return word_vector
    
    def get_multiple_word_vectors(self, model, data_loader, words: list) -> Dict[str, np.ndarray]:
        """여러 단어의 벡터를 한 번에 반환"""
        import numpy as np
        
        vectors = {}
        model.eval()
        
        with torch.no_grad():
            for word in words:
                if word in data_loader.word2id:
                    word_id = data_loader.word2id[word]
                    vectors[word] = model.u_embeddings.weight[word_id].cpu().numpy()
                else:
                    print(f"Warning: '{word}' not found in vocabulary")
                    vectors[word] = None
        
        return vectors
    
    def get_all_vectors(self, model, data_loader) -> Dict[str, np.ndarray]:
        """모든 단어의 벡터를 반환 (메모리 주의)"""
        import numpy as np
        
        model.eval()
        vectors = {}
        
        with torch.no_grad():
            all_embeddings = model.u_embeddings.weight.cpu().numpy()
            
            for word_id, word in data_loader.id2word.items():
                vectors[word] = all_embeddings[word_id]
        
        return vectors
    
    def _find_similar_words(self, model, data_loader, target_word: str, top_k: int):
        """특정 단어와 유사한 단어들 찾기"""
        import torch.nn.functional as F
        
        if target_word not in data_loader.word2id:
            return []
        
        target_id = data_loader.word2id[target_word]
        target_embedding = model.u_embeddings.weight[target_id].unsqueeze(0)
        
        # 모든 단어 임베딩과 코사인 유사도 계산
        all_embeddings = model.u_embeddings.weight
        similarities = F.cosine_similarity(target_embedding, all_embeddings, dim=1)
        
        # 자기 자신 제외하고 상위 k개 추출
        similarities[target_id] = -1  # 자기 자신 제외
        top_indices = similarities.topk(top_k).indices
        
        similar_words = []
        for idx in top_indices:
            word = data_loader.id2word[idx.item()]
            similarity = similarities[idx].item()
            similar_words.append((word, similarity))
        
        return similar_words
