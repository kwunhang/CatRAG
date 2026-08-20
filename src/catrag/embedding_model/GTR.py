from typing import List, Optional, Union, Dict, Any
from copy import deepcopy

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

class GTRModel(BaseEmbeddingModel):
    """
    Wrapper for GTR (Generalizable T5-based Retriever) models using SentenceTransformers.
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = "sentence-transformers/gtr-t5-base") -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name:
            self.embedding_model_name = embedding_model_name

        logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        # Initialize SentenceTransformer
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            **self.embedding_config.model_init_params
        )
        
        # Set max seq length if defined in global config
        if hasattr(self.global_config, 'embedding_max_seq_len') and self.global_config.embedding_max_seq_len:
             self.embedding_model.max_seq_length = self.global_config.embedding_max_seq_len

        self.embedding_model.eval()
        
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.
        """
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "trust_remote_code": True,
            },
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": self.global_config.embedding_return_as_normalized
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        """
        if isinstance(texts, str): 
            texts = [texts]

        # Merge default config params with any runtime kwargs
        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: 
            params.update(kwargs)

        logger.debug(f"Calling {self.__class__.__name__} batch_encode with params: {params}")

        embeddings = self.embedding_model.encode(texts)

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        elif isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        results = embeddings
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
