import numpy as np
from typing import Dict, List, Optional, Tuple

from  entities import SkipGramModel # skipgram을 SkipGramModel로 가정
from .DataLoader import MemoryDataLoader, MemoryWord2vecDataset
from .Trainer import Word2VecTrainer
from ..Document import DocumentService


class Word2VecService:
    """Word2Vec 서비스 - 전체 Word2Vec 파이프라인 관리"""
    
    def __init__(self,
                doc: DocumentService,
                model: SkipGramModel = None,
                trainer: Word2VecTrainer = None,
                dataset: MemoryWord2vecDataset = None,
                data_loader: MemoryDataLoader = None):
        
        self.doc = doc
        self.model = model
        self.trainer = trainer
        self.dataset = dataset
        self.data_loader = data_loader
        
        # 편의를 위해 문장 인덱스도 저장
        self.word_indices = doc.get_sentences_with_word2id()
        
        # 훈련 상태 추적
        self.is_trained = False
        self.training_history = []
    
    @classmethod
    def create_default(cls, doc_service: DocumentService, min_count: int = 1) -> 'Word2VecService':
        """기본 설정으로 Word2Vec 서비스 생성"""

        # 문서 서비스에서 필요한 데이터 추출
        word_data = doc_service.get_word2vec_data(min_count=min_count) # 이제 0-indexed ID를 반환
        
        # DocumentService에서 원래 문장 인덱스를 가져오는 대신,
        # 직접 단어 콘텐츠를 사용하여 새로운 0-indexed ID로 매핑
        # 이렇게 하면 DocumentService의 원본 ID와 Word2Vec 모델의 0-indexed ID 간의 불일치 문제 해결
        preprocessed_sentences = doc_service.get_preprocessed_sentences() # 단어 텍스트 리스트
        new_word2id_map = word_data['word2id'] # DocumentService에서 0-indexed로 생성된 맵

        sentences_with_indices = []
        for sentence_words in preprocessed_sentences:
            remapped_sentence = []
            for word_content in sentence_words:
                # 단어 콘텐츠를 사용하여 새로운 0-indexed ID로 변환
                if word_content in new_word2id_map:
                    remapped_sentence.append(new_word2id_map[word_content])
            if remapped_sentence: # 빈 문장은 제외
                sentences_with_indices.append(remapped_sentence)

        print(f"Creating Word2Vec service with vocabulary size: {word_data['vocab_size']}")
        
        # 모델 생성
        model = SkipGramModel(word_data['vocab_size'], emb_dimension=300)

        # 데이터 로더 생성
        data_loader = MemoryDataLoader(
            sentences=sentences_with_indices,
            word2id=word_data['word2id'],
            id2word=word_data['id2word'],
            word_frequency=word_data['word_frequency']
        )

        # 데이터셋 생성
        dataset = MemoryWord2vecDataset(data_loader, window_size=5)

        # 트레이너 생성
        trainer = Word2VecTrainer(iterations=10, initial_lr=0.025, batch_size=300)
        
        return cls(doc_service, model, trainer, dataset, data_loader)
    
    @classmethod
    def create_custom(cls, 
                    doc_service: DocumentService,
                    embedding_dim: int = 100,
                    window_size: int = 5,
                    iterations: int = 30,
                    learning_rate: float = 0.001,
                    batch_size: int = 32,
                    min_count: int = 5) -> 'Word2VecService':
        """커스텀 설정으로 Word2Vec 서비스 생성"""
        
        word_data = doc_service.get_word2vec_data(min_count=min_count) # 이제 0-indexed ID를 반환
        
        preprocessed_sentences = doc_service.get_preprocessed_sentences() # 단어 텍스트 리스트
        new_word2id_map = word_data['word2id'] # DocumentService에서 0-indexed로 생성된 맵

        sentences_with_indices = []
        for sentence_words in preprocessed_sentences:
            remapped_sentence = []
            for word_content in sentence_words:
                if word_content in new_word2id_map:
                    remapped_sentence.append(new_word2id_map[word_content])
            if remapped_sentence: # 빈 문장은 제외
                sentences_with_indices.append(remapped_sentence)
        
        model = SkipGramModel(word_data['vocab_size'], emb_dimension=embedding_dim)
        
        data_loader = MemoryDataLoader(
            sentences=sentences_with_indices,
            word2id=word_data['word2id'],
            id2word=word_data['id2word'],
            word_frequency=word_data['word_frequency'],
            min_count=min_count
        )
        
        dataset = MemoryWord2vecDataset(data_loader, window_size=window_size)
        trainer = Word2VecTrainer(
            iterations=iterations, 
            initial_lr=learning_rate, 
            batch_size=batch_size
        )
        
        return cls(doc_service, model, trainer, dataset, data_loader)
    
    def train(self, output_file: str = "word2vec_embeddings.txt") -> 'Word2VecService':
        """모델 훈련 실행"""
        print("Starting Word2Vec training...")
        
        self.model = self.trainer.train(
            model=self.model,
            dataset=self.dataset,
            data_loader=self.data_loader,
            output_file=output_file
        )
        
        self.is_trained = True
        print(f"Training completed! Embeddings saved to {output_file}")
        
        return self  # 메서드 체이닝을 위해 self 반환
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """단어의 numpy 벡터 반환"""
        if not self.is_trained:
            print("Warning: Model is not trained yet. Call train() first.")
        
        return self.trainer.get_word_vector(self.model, self.data_loader, word)
    
    def get_multiple_word_vectors(self, words: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """여러 단어의 벡터를 한 번에 반환"""
        if not self.is_trained:
            print("Warning: Model is not trained yet. Call train() first.")
        
        return self.trainer.get_multiple_word_vectors(self.model, self.data_loader, words)
    
    def get_all_vectors(self) -> Dict[str, np.ndarray]:
        """모든 단어의 벡터를 반환 (메모리 주의)"""
        if not self.is_trained:
            print("Warning: Model is not trained yet. Call train() first.")
        
        return self.trainer.get_all_vectors(self.model, self.data_loader)
    
    def find_similar_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """유사한 단어들 찾기"""
        if not self.is_trained:
            print("Warning: Model is not trained yet. Call train() first.")
            return []
        
        return self.trainer._find_similar_words(self.model, self.data_loader, word, top_k)
    
    def evaluate_similarity(self, test_words: Optional[List[str]] = None, top_k: int = 5):
        """단어 유사도 평가"""
        if not self.is_trained:
            print("Warning: Model is not trained yet. Call train() first.")
            return
        
        self.trainer.evaluate_similarity(self.model, self.data_loader, test_words, top_k)
    
    def word_analogy(self, word_a: str, word_b: str, word_c: str, top_k: int = 1) -> List[Tuple[str, float]]:
        """단어 유추: word_a is to word_b as word_c is to ?
        예: king is to man as queen is to woman
        """
        if not self.is_trained:
            print("Warning: Model is not trained yet. Call train() first.")
            return []
        
        # 필요한 단어들이 어휘에 있는지 확인
        missing_words = []
        for word in [word_a, word_b, word_c]:
            if word not in self.data_loader.word2id:
                missing_words.append(word)
        
        if missing_words:
            print(f"Words not in vocabulary: {missing_words}")
            return []
        
        # 벡터 연산: word_b - word_a + word_c
        vec_a = self.get_word_vector(word_a)
        vec_b = self.get_word_vector(word_b)
        vec_c = self.get_word_vector(word_c)
        
        target_vector = vec_b - vec_a + vec_c
        
        # 가장 유사한 단어 찾기 (입력 단어들 제외)
        similarities = []
        exclude_words = {word_a, word_b, word_c}
        
        import torch.nn.functional as F
        import torch
        
        self.model.eval()
        with torch.no_grad():
            target_tensor = torch.FloatTensor(target_vector).unsqueeze(0)
            all_embeddings = self.model.u_embeddings.weight
            
            # 코사인 유사도 계산
            cosine_sims = F.cosine_similarity(target_tensor, all_embeddings, dim=1)
            
            # 상위 결과 중에서 입력 단어들 제외
            top_indices = cosine_sims.topk(top_k + len(exclude_words)).indices
            
            for idx in top_indices:
                word = self.data_loader.id2word[idx.item()]
                if word not in exclude_words:
                    similarity = cosine_sims[idx].item()
                    similarities.append((word, similarity))
                    if len(similarities) >= top_k:
                        break
        
        return similarities
    
    def get_vocabulary_info(self) -> Dict[str, any]:
        """어휘 정보 반환"""
        return {
            'vocab_size': len(self.data_loader.word2id),
            'total_sentences': len(self.word_indices),
            'total_tokens': self.data_loader.token_count,
            'embedding_dimension': self.model.emb_dimension,
            'most_frequent_words': self._get_most_frequent_words(10),
            'is_trained': self.is_trained
        }
    
    def _get_most_frequent_words(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """빈도 상위 단어들 반환"""
        word_freq_pairs = []
        for word_id, freq in self.data_loader.word_frequency.items():
            word = self.data_loader.id2word[word_id]
            word_freq_pairs.append((word, freq))
        
        return sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)[:top_n]
    
    def save_model(self, filepath: str):
        """모델 저장"""
        if not self.is_trained:
            # print("Warning: Model is not trained yet.")
            return
        
        import torch
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'word2id': self.data_loader.word2id,
            'id2word': self.data_loader.id2word,
            'word_frequency': self.data_loader.word_frequency,
            'vocab_size': len(self.data_loader.word2id),
            'embedding_dim': self.model.emb_dimension
        }, filepath)
        
        # print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        import torch
        
        checkpoint = torch.load(filepath, map_location=self.trainer.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 데이터 로더 정보도 업데이트 (필요한 경우)
        self.is_trained = True
        # print(f"Model loaded from {filepath}")
    
    def __str__(self) -> str:
        """서비스 정보 문자열 표현"""
        info = self.get_vocabulary_info()
        return f"""
Word2VecService Status:
- Vocabulary Size: {info['vocab_size']}
- Total Sentences: {info['total_sentences']}
- Total Tokens: {info['total_tokens']}
- Embedding Dimension: {info['embedding_dimension']}
- Trained: {info['is_trained']}
- Most Frequent Words: {', '.join([word for word, _ in info['most_frequent_words'][:5]])}
"""
