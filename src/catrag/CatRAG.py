import ast
import json
import os
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time
import copy
# SAFETY: Pickle is used only for CatRAG-generated caches inside a trusted
# save_dir. Never load a save_dir supplied by an untrusted party. See README.
import pickle
import concurrent.futures

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .evaluation.retrieval_eval import RetrievalRecall, ChainRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score, QAAccuracy, QAJSRScore
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig


logger = logging.getLogger(__name__)
SKIPPED_EDGE_WEIGHT = 0.3 # assigned weight as Weak entity refer to LLM Score Projection


class CatRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific CatRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"CatRAG init with config:\n  {_print_config}\n")

        # LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")
        self.query_to_embedding_store = os.path.join(self.working_dir, f"query_to_embedding.pkl")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)
        
        if self.global_config.qa_llm_model is not None and self.global_config.qa_llm_base_url is not None: 
            qa_llm_config = copy.deepcopy(self.global_config)
            qa_llm_config.llm_name = self.global_config.qa_llm_model
            qa_llm_config.llm_base_url = self.global_config.qa_llm_base_url
            self.qa_llm_model: BaseLLM = _get_llm_class(qa_llm_config)
            logging.info(f"Using another LLM for QA reading task: '{self.global_config.qa_llm_base_url}'; '{self.global_config.qa_llm_base_url}' ")
        else:
            self.qa_llm_model: BaseLLM = self.llm_model
        
        # Score LLM for edge weight adjust, use the same LLM here. Smaller model can be use, to lower the latency and consumption
        self.score_llm_model: BaseLLM = self.llm_model
        
        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
            self.openie = VLLMOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0
        self.llm_score_time = 0
        self.passage_reranking_time = 0

        self.ent_node_to_chunk_ids = None

        self.ent_node_to_fact_ids = None
        
        self.node_abs_store = None

    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"directed_graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)
                # Change the graph to directed if it is undirected
                if not preloaded_graph.is_directed():
                    preloaded_graph.to_directed(mode="mutual")


        if preloaded_graph is None:
            # Create the graph as directed graph
            return ig.Graph(directed=True)

        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        if self.global_config.save_openie:
            raise RuntimeError(
                "Offline OpenIE completed and its results were saved. Run "
                "indexing again with openie_mode='online' and the same "
                "save_dir and llm_name to build the graph."
            )

        raise RuntimeError(
            "Offline OpenIE completed, but save_openie=False prevented its "
            "results from being saved. Run the offline stage again with "
            "save_openie=True before starting online graph construction."
        )

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the CatRAG framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")
        
        
        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict)

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}
        self.ent_node_to_fact_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()

        logger.info("Start Build summary")
        self.build_graph_node_sum()
        logger.info("End Build summary")

    def build_graph_node_sum(self):
        """
        Builds summaries for each entity node in the graph by providing the node's neighbors and facts.
        The summaries will be used in retrieval to assist LLM in determining relevant edge weights around seed nodes.
        """
        
        # try to load the exist built summary store
        summary_file_name = os.path.join(self.global_config.cache_dir, f"node_summaries_final_{self.global_config.dataset}.json")
        if os.path.exists(summary_file_name):
            with open(summary_file_name, 'r', encoding='utf-8') as f:
                loaded_summaries = json.load(f)
                self.node_abs_store = loaded_summaries
            logger.info(
                f"Loaded summary from {summary_file_name} with {len(self.node_abs_store)} node summaries"
            )
            return
        
        self.node_abs_store = {}  # Initialize the summary store
        
        # Get all entity nodes
        entity_nodes_idx = [(idx, node) for idx, node in enumerate(self.graph.vs) 
                        if node['name'].startswith("entity-")]
        
        # Process in batches
        batch_size = 50
        all_batches = [entity_nodes_idx[i:i + batch_size] 
                    for i in range(0, len(entity_nodes_idx), batch_size)]
        
        logger.info(f"Processing {len(entity_nodes_idx)} entity nodes in {len(all_batches)} batches")
        for batch_num, batch in enumerate(all_batches):
            logger.info(f"Processing batch {batch_num + 1}/{len(all_batches)}")
            
            cur_messages = []
            batch_nodes_info = []
            
            # Prepare messages for each node in the current batch
            for idx, node in batch:
                try:
                    # Get entity information
                    entity_row = self.entity_embedding_store.get_row(node['name'])
                    if not entity_row:
                        logger.warning(f"Entity row not found for node: {node['name']}")
                        continue
                    
                    # Get fact triplets connected to this entity
                    fact_ids_set = self.ent_node_to_fact_ids.get(node['name'], set())
                    
                    if len(fact_ids_set) <= 3:
                        logger.info("skip small entity node first")
                        continue
                    
                    fact_rows_dict = self.fact_embedding_store.get_rows(fact_ids_set)
                    fact_triplets = []
                    
                    for fact_id in fact_ids_set:
                        if fact_id in fact_rows_dict:
                            try:
                                triple = tuple(ast.literal_eval(fact_rows_dict[fact_id]["content"]))
                                fact_triplets.append(str(triple))
                            except (SyntaxError, ValueError, TypeError):
                                logger.warning(f"Failed to parse fact: {fact_rows_dict[fact_id]['content']}")
                                continue
                    
                    # Format triplets for the prompt
                    prompt_triplets = "{\n" + "\n".join(fact_triplets) + "\n}" if fact_triplets else "No connected facts"
                    
                    # Check if prompt template exists
                    if not self.prompt_template_manager.is_template_name_valid(name='node_summarize'):
                        logger.error("Does not have a customized prompt template to summarize node.")
                        import sys
                        sys.exit(1)
                    
                    message = self.prompt_template_manager.render(
                        name='node_summarize', 
                        summary_length="150",
                        language="English", 
                        entity=entity_row['content'], 
                        fact_triplets=prompt_triplets
                    )
                    
                    cur_messages.append(message)
                    batch_nodes_info.append({
                        'node_idx': idx,
                        'node_name': node['name'],
                        'entity_content': entity_row['content'],
                        'fact_count': len(fact_triplets)
                    })
                    
                except Exception as e:
                    logger.error(f"Error preparing message for node {node['name']}: {str(e)}")
                    continue
            
            # Skip if no valid messages in this batch
            if not cur_messages:
                logger.info(f"No sumary is construct in batch {batch_num + 1}")
                continue
            
            # Process LLM calls for the current batch
            cur_response_messages = []
            cur_metadata = []
            cur_cache_hit = []
            
            batch_results = self.llm_model.batch_infer(cur_messages)
            # Process results
            for i, (response_message, metadata, cache_hit) in enumerate(batch_results):
                cur_response_messages.append(response_message)
                cur_metadata.append(metadata)
                cur_cache_hit.append(cache_hit)
            
            
            # Store results for this batch
            for i, (node_info, response_msg) in enumerate(zip(batch_nodes_info, cur_response_messages)):
                if response_msg and not response_msg.startswith("[ANSWER SKIPPED]"):
                    try:
                        llm_ans = response_msg.split('Answer:')[1].strip()
                    except Exception as e:
                        llm_ans = response_msg.strip()
                    self.node_abs_store[node_info['node_name']] = {
                        'node_idx': node_info['node_idx'],
                        'entity': node_info['entity_content'],
                        'summary': llm_ans,
                        'fact_count': node_info['fact_count'],
                        'metadata': cur_metadata[i] if i < len(cur_metadata) else {},
                        'cache_hit': cur_cache_hit[i] if i < len(cur_cache_hit) else False
                    }
                    logger.debug(f"Stored summary for node: {node_info['node_name']}")
                else:
                    logger.warning(f"Skipped storing summary for node: {node_info['node_name']}")
            if (batch_num%1) == 0 :
                self._save_node_summaries()
        
        successful_nodes = len(self.node_abs_store)
        logger.info(f"Successfully generated summaries for {successful_nodes}/{len(entity_nodes_idx)} entity nodes")
        
        # Save summaries after all summary is built
        self._save_node_summaries(final=True)
        
    def _save_node_summaries(self, final = False):
        """Save node summaries to disk for persistence"""
        post_fix = "_final" if final else ""
        try:
            summary_file = os.path.join(self.global_config.cache_dir, f"node_summaries{post_fix}_{self.global_config.dataset}.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                # Convert to serializable format
                serializable_summaries = {}
                for node_name, summary_data in self.node_abs_store.items():
                    serializable_summaries[node_name] = {
                        'node_idx': summary_data['node_idx'],
                        'entity': summary_data['entity'],
                        'summary': summary_data['summary'],
                        'fact_count': summary_data['fact_count']
                    }
                json.dump(serializable_summaries, f, ensure_ascii=False, indent=2)
            logger.info(f"Node summaries saved to {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save node summaries: {str(e)}")

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the CatRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        # Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        # Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        # Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        # Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        # Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node].keys()

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        # Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete) + list(chunk_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def ner_query(self, query:str):
        query_ner_embeddings = []
        ner_node_list = set()
        ner_prompt = self.prompt_template_manager.render(name=f'ner_query', query=query)
        try:
            llm_result = self.llm_model.infer(ner_prompt)
            response_message, metadata, cache_hit = llm_result
            if cache_hit:
                logging.info("cache_hit in NER")
            message = re.sub('```json', ' ', response_message)
            message = re.sub('```', ' ', message)
            parsed_response = json.loads(message)
            query_ner_list = parsed_response.get('named_entities', [])
            if not isinstance(query_ner_list, list) or not all(
                isinstance(entity, str) for entity in query_ner_list
            ):
                raise ValueError("'named_entities' must be a list of strings")
            query_ner_list = [text_processing(p) for p in query_ner_list]
            
            query_ner_embeddings = self.embedding_model.batch_encode(query_ner_list,
                                                                    instruction=get_query_instruction('ner_to_node'),
                                                                    norm=True)
        except Exception as e:
            logging.warning(f"error in Ner to node: {str(e)}")
        for embedding in query_ner_embeddings:
            try:
                ner_node_scores = np.dot(self.entity_embeddings, embedding.T) # shape: (#facts, )
                ner_node_scores = np.squeeze(ner_node_scores) if ner_node_scores.ndim == 2 else ner_node_scores
                phrase_id = np.argsort(ner_node_scores)[-1:][::-1].tolist()
                ner_node_list.add(phrase_id[0])
            except Exception as e:
                logger.error(f"Error computing fact scores: {str(e)}")
        return list(ner_node_list)
    
    
    def get_top_fact_rerank(self, query:str):
        query_fact_scores = self.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
        return query_fact_scores, top_k_fact_indices, top_k_facts, rerank_log
    
    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the CatRAG framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
            chain_recall_evaluator = ChainRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            rerank_start = time.time()
            # Found seed (Query to triplet) and weak-seed (NER to node) in parallel 
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_get_fact_rerank = executor.submit(
                    self.get_top_fact_rerank,
                    query
                )
                future_ner = executor.submit(
                    self.ner_query,
                    query
                )
                
                query_fact_scores, top_k_fact_indices, top_k_facts, _ = future_get_fact_rerank.result()
                ner_node_list = future_ner.result()
            rerank_end = time.time()

            self.rerank_time += rerank_end - rerank_start

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                        link_top_k=self.global_config.linking_top_k,
                                                                                        query_fact_scores=query_fact_scores,
                                                                                        top_k_facts=top_k_facts,
                                                                                        top_k_fact_indices=top_k_fact_indices,
                                                                                        ner_node_indices=ner_node_list,
                                                                                        passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))


        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")
        
        # logging for extra function
        logger.info(f"Total LLM scoring Time {self.llm_score_time:.2f}s")
        logger.info(f"Total Passage Reranking Time {self.passage_reranking_time:.2f}s")
        
        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            # Evaluating chain recall
            chain_recall_result, example_retrieval_results = chain_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=[2,5])
            logger.info(f"Evaluation result for chain recall: {chain_recall_result}")
            overall_retrieval_result.update(chain_recall_result)

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the CatRAG framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores, accuracy, JSR scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)
            qa_acc_evaluator = QAAccuracy(global_config=self.global_config)

        if gold_answers is not None and gold_docs is not None:
            qa_jsr_evaluator = QAJSRScore(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_acc_result, example_qa_acc_results = qa_acc_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions])

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            
            # jsr score
            overall_qa_em_result.update(overall_qa_acc_result) 
            if gold_docs is not None and isinstance(queries[0], QuerySolution):
                overall_qa_jsr_result, example_qa_jsr_results = qa_jsr_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                    gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs[:5] for retrieval_result in queries])
                overall_qa_em_result.update(overall_qa_jsr_result)

            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)
            chain_recall_evaluator = ChainRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
            
            # Evaluating chain recall
            chain_recall_result, example_retrieval_results = chain_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in queries], k_list=[2,5])
            logger.info(f"Evaluation result for chain recall: {chain_recall_result}")
            overall_retrieval_result.update(chain_recall_result)

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match, F1 scores, accuracy, JSR scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)
            qa_acc_evaluator = QAAccuracy(global_config=self.global_config)

        if gold_answers is not None and gold_docs is not None:
            qa_jsr_evaluator = QAJSRScore(global_config=self.global_config)
            
        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)
        
        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_acc_result, example_qa_acc_results = qa_acc_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions])

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_em_result.update(overall_qa_acc_result) 
            if gold_docs is not None and isinstance(queries[0], QuerySolution):
                overall_qa_jsr_result, example_qa_jsr_results = qa_jsr_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                    gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs[:5] for retrieval_result in queries])
                overall_qa_em_result.update(overall_qa_jsr_result)

            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        # Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            if self.global_config.dataset.startswith("hover"):
                prompt_user += query_solution.question + '\nThought: '
            else:
                prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        all_response_message = []
        all_metadata = []
        all_cache_hit = []

        batch_size = 16
        pbar = tqdm(total=len(all_qa_messages), desc="Batch QA")
        for start in range(0, len(all_qa_messages), batch_size):
            batch_results = self.qa_llm_model.batch_infer(all_qa_messages[start: start+batch_size], max_workers=batch_size)
            for result in batch_results:
                response_message, metadata, cache_hit = result
                all_response_message.append(response_message)
                all_metadata.append(metadata)
                all_cache_hit.append(cache_hit)
            pbar.update(batch_size)
        pbar.close()

        # Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple], !?:List[List[Tuple]]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix="entity-")
                    node_2_key = compute_mdhash_id(content=triple[2], prefix="entity-")

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 2

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)
                    
                    # Record the fact that link to the phrase node
                    fact_id = compute_mdhash_id(str(triple), prefix="fact-")
                    self.ent_node_to_fact_ids[node_key] = self.ent_node_to_fact_ids.get(node_key, set()).union(set([fact_id]))
                    self.ent_node_to_fact_ids[node_2_key] = self.ent_node_to_fact_ids.get(node_2_key, set()).union(set([fact_id]))
                    
                    # Record the passage that link with phrase node. And record number of fact between passage and phrase node 
                    for node in [node_key,node_2_key]:
                        ent_node_chunk_dict = self.ent_node_to_chunk_ids.get(node, dict())
                        ent_node_chunk_dict[chunk_key] = ent_node_chunk_dict.get(chunk_key, set()).union(set([fact_id]))
                        self.ent_node_to_chunk_ids[node] = ent_node_chunk_dict

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Adds edges connecting passage nodes to phrase nodes in the graph.

        This method is responsible for iterating through a list of chunk identifiers
        and their corresponding triple entities. It calculates and adds new edges
        between the passage nodes (defined by the chunk identifiers) and the phrase
        nodes (defined by the computed unique hash IDs of triple entities). The method
        also updates the node-to-node statistics map and keeps count of newly added
        passage nodes.

        Parameters:
            chunk_ids : List[str]
                A list of identifiers representing passage nodes in the graph.
            chunk_triple_entities : List[List[str]]
                A list of lists where each sublist contains entities (strings) associated
                with the corresponding chunk in the chunk_ids list.

        Returns:
            int
                The number of new passage nodes added to the graph.
        """

        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score * 3  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new nodes to the graph from entity and passage embedding stores based on their attributes.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity embedding store and the passage
        embedding store. The method checks attributes and ensures no duplicates are added.
        New nodes are prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                # Add two direction of edge weight
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
                valid_edges.append((target_node_id, source_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the graph such as the number of nodes,
        triples, and their classifications.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase and
        passage nodes, total nodes, extracted triples, triples involving passage
        nodes, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_passage_nodes: The number of unique passage nodes.
                - num_total_nodes: The total number of nodes (sum of phrase and passage nodes).
                - num_extracted_triples: The number of unique extracted triples.
                - num_triples_with_passage_node: The number of triples involving at least one
                  passage node.
                - num_synonymy_triples: The number of synonymy triples (distinct from extracted
                  triples and those with passage nodes).
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of passage nodes
        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        num_triples_with_passage_node = 0
        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.

        Security note: graph and query-embedding pickle files are local CatRAG
        cache artifacts. The configured save_dir must be trusted; never load
        cache files obtained from an untrusted source.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        # Store the embedding to reduce cost
        if os.path.exists(self.query_to_embedding_store) and self.global_config.overhead_bypass_cache==False:
            with open(self.query_to_embedding_store, 'rb') as inp:
                self.query_to_embedding = pickle.load(inp)
            logger.info("Restore query embedding.")
        else:
            self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids()) # a list of passage node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            # If the graph is empty but we have nodes, we need to add them
            if actual_node_count == 0 and expected_node_count > 0:
                logger.info(f"Initializing graph with {expected_node_count} nodes")
                self.add_new_nodes()
                self.save_igraph()

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                self.add_new_nodes()
                self.save_igraph()
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.passage_node_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: passage_node_keys={len(self.passage_node_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.passage_node_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.passage_node_keys]

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.passage_node_keys, chunk_triples)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding
        
        with open(self.query_to_embedding_store, 'wb') as outp:
            pickle.dump(self.query_to_embedding, outp, pickle.HIGHEST_PROTOCOL)
        logger.info("Save query embedding.")

    
    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        ner_node_indices: Optional[List[int]],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            ner_node_indices (Optional[List[int]]): Corresponding graph indices for the weak phrase
                seed node from NER.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()
        phrase_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, {})) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                phrases_and_ids.add((phrase, phrase_id))
                phrase_ids.add(phrase_id)

        phrase_weights /= number_of_occurs

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)

        modified_edges = {}
        # NOTE: adjust the seed entity->passage edge weight, phrase with seed triplet with high weight
        top_k_facts_id = set()
        for triple in top_k_facts:
            fact_id = compute_mdhash_id(str(triple), prefix="fact-")
            top_k_facts_id.add(fact_id)

        for phrase, phrase_id in phrases_and_ids:
            node_key = compute_mdhash_id(content=phrase, prefix=("entity-"))
            ent_node_chunk_dict = self.ent_node_to_chunk_ids.get(node_key, {})
            
            if not ent_node_chunk_dict:
                continue

            # 1. Get all edges starting from this entity node
            # We map target_vertex_idx -> edge_object for O(1) lookup later
            all_out_edges = self.graph.es.select(_source=node_key)
            target_to_edge = {edge.target: edge for edge in all_out_edges}

            # Lists to store the data for calculation before updating
            passage_edges_data = [] 
            total_original_passage_weight = 0.0

            # 2. Identify Passage Edges and Calculate "Raw" Boosted Weights
            has_overlap_boost = False

            for chunk_key, ent_chunk_facts in ent_node_chunk_dict.items():
                # Find the specific edge connecting entity to this chunk
                target_idx = self.node_name_to_vertex_idx.get(chunk_key)
                if target_idx is None or target_idx not in target_to_edge:
                    # logging.warning(f"Edge not found for chunk {chunk_key}")
                    continue
                
                edge = target_to_edge[target_idx]
                current_weight = edge["weight"]
                total_original_passage_weight += current_weight

                # Check for fact overlap
                fact_embedding_idxs = set(ent_chunk_facts)
                overlap = fact_embedding_idxs & top_k_facts_id
                
                # Apply the logic: 2 if overlap, 1 if not (relative to current weight)
                if len(overlap) > 0:
                    raw_new_weight = current_weight * 2.5
                    has_overlap_boost = True
                else:
                    raw_new_weight = current_weight 
                
                passage_edges_data.append({
                    "edge": edge,
                    "raw_new_weight": raw_new_weight
                })

            # 3. Normalize and Update
            # Only proceed if we actually found edges and at least one had an overlap
            if passage_edges_data and has_overlap_boost:
                total_raw_new_weight = sum(d["raw_new_weight"] for d in passage_edges_data)
                
                if total_raw_new_weight > 0:
                    for item in passage_edges_data:
                        edge = item["edge"]
                        raw_val = item["raw_new_weight"]
                        
                        normalized_weight = (raw_val / total_raw_new_weight) * total_original_passage_weight
                        
                        # Store original for rollback
                        if edge.index not in modified_edges:
                            modified_edges[edge.index] = edge["weight"]
                        
                        edge.update_attributes({"weight": normalized_weight})
        # End of adjust the seed entity->passage edge weight,
        
        # Add weak seed from NER (Symbolic Anchoring)
        for indice in ner_node_indices:
            # only add node if it is not in original seed indice set 
            try:
                if phrase_weights[indice]== 0:
                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score = self.global_config.weak_weight / len(self.ent_node_to_chunk_ids[phrase_key])
                    else:
                        weighted_fact_score = self.global_config.weak_weight
                    phrase_weights[indice] += weighted_fact_score
                    phrase_key = self.graph.vs[indice]['name']
                    row = self.entity_embedding_store.get_row(phrase_key)  
                    phrase_content = row['content']
                    linking_score_map[phrase_content] = weighted_fact_score
            except Exception as e:
                logging.warning(f"Add NER meet error: {str(e)}")
        
        # LLM score only TOP 5 for more fair evaluaion set up
        sorted_candidates = sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)
        linking_score_map = dict(sorted_candidates[:5])
        # End of Add weak seed from NER (Symbolic Anchoring)
        
        # Dynamic adjust edge weight by LLM (Dynamic Edge Weighting)
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        assert query_embedding is not None, f"Query embedding not found for query: {query}"

        # Batch LLM call
        cur_score_update_dict = {}
        neighbors_key_all_phrase = []
        entity_all_phrase = []
        skipped_neighbor_indices = set()

        for phrase, phrase_score in linking_score_map.items():
            phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
            if phrase_key not in self.node_name_to_vertex_idx: continue

            phrase_idx = self.node_name_to_vertex_idx[phrase_key]
            neighbors_node = self.graph.vs[phrase_idx].successors()
            
            seed_node_fact_ids = self.ent_node_to_fact_ids.get(phrase_key, set())
            
            # 1. Identify valid neighbors (Structural Filter Only)
            valid_neighbors = [] # List of tuple: (neighbor_name, shared_fact_ids)
            
            for node in neighbors_node:
                n_name = node["name"]
                if not n_name.startswith("entity-"): continue
                
                neighbor_fact_ids = self.ent_node_to_fact_ids.get(n_name, set())
                shared_facts = seed_node_fact_ids.intersection(neighbor_fact_ids)
                
                if len(shared_facts) > 0:
                    valid_neighbors.append((n_name, shared_facts))

            # 2. Apply Vector Filter if number of neighbour > llm_score_max_edge (Coarse-Grained Candidate Pruning)
            final_neighbor_keys = []

            if len(valid_neighbors) <= self.global_config.llm_score_max_edge:
                final_neighbor_keys = [vn[0] for vn in valid_neighbors]
            else:
                scored_candidates = []
                
                for n_name, shared_facts in valid_neighbors:
                    max_vec_score = -1.0
                    for fid in shared_facts:
                        f_idx = self.fact_embedding_store.hash_id_to_idx.get(fid) 
                        if f_idx is not None and f_idx < len(query_fact_scores):
                            score = query_fact_scores[f_idx]
                            if score > max_vec_score:
                                max_vec_score = score
                    
                    if max_vec_score > -1.0:
                        scored_candidates.append((n_name, max_vec_score))
                
                # Sort descending by vector similarity and Slice top-K
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                final_neighbor_keys = [x[0] for x in scored_candidates[:self.global_config.llm_score_max_edge]]
                skipped = scored_candidates[self.global_config.llm_score_max_edge:]
                for skip_name, _ in skipped:
                        skip_idx = self.node_name_to_vertex_idx.get(skip_name)
                        if skip_idx is not None:
                            skipped_neighbor_indices.add(skip_idx)

            # 3. Batching for LLM (Standard Logic)
            if not final_neighbor_keys:
                continue

            if len(final_neighbor_keys) > self.global_config.llm_score_batch:
                chunk_fact_rows = [final_neighbor_keys[i:i+self.global_config.llm_score_batch] for i in range(0, len(final_neighbor_keys), self.global_config.llm_score_batch)]
                neighbors_key_all_phrase.extend(chunk_fact_rows)
                entity_all_phrase.extend([phrase] * len(chunk_fact_rows)) 
            else:
                neighbors_key_all_phrase.append(final_neighbor_keys)
                entity_all_phrase.append(phrase)

        llm_score_start_time = time.time()
        neighbor_scores_all_phrase, _ = self.call_llm_score(query, neighbors_key_all_phrase, entity_all_phrase, top_k_facts)
        llm_score_end_time = time.time()
        
        self.llm_score_time += llm_score_end_time - llm_score_start_time
        
        for neighbor_scores in neighbor_scores_all_phrase:
            for cur_item in neighbor_scores:
                entity, score = cur_item["entity"], cur_item["score"]
                if score <= 3:
                    # the triplet is irrelvant
                    score = 0
                elif score <=6: # weak link (4-6), the edge weight would in range of 0.2~0.3
                    score = score / 20
                elif score <10: # strong link (7-9), the edge weight would in range of 2~3
                    score = (score-3) / 2
                elif score == 10: # 
                    score = 5
                else: # score large than 10,have error, assign 10 first
                    score = 5
                    
                node_key = compute_mdhash_id(content=entity, prefix=("entity-"))
                target_node = node_key
                # assign the max score of relations as edge weight
                cur_score_update_dict[target_node] = max(cur_score_update_dict.get(target_node, 0.0), score)
                
        cur_score_idx_dict = {self.node_name_to_vertex_idx[node_name]: score*2 for node_name, score in cur_score_update_dict.items()}
        
        for phrase, _ in linking_score_map.items():
            phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
            edges = self.graph.es.select(_source=phrase_key)
            check = False
            for edge in edges:
            # which mean the edge is from chunk triple and the score need to update. Update time the number of tuplet occur
                if edge.target in cur_score_idx_dict:
                    if edge.index not in modified_edges:
                        modified_edges[edge.index] = edge["weight"]
                        
                    edge.update_attributes({"weight": cur_score_idx_dict[edge.target] * edge["weight"]})      
                    check=True 
                elif edge.target in skipped_neighbor_indices:
                    # Assign low weight as weak neighbour
                    if edge.index not in modified_edges:
                        modified_edges[edge.index] = edge["weight"]
                        
                    edge.update_attributes({"weight": SKIPPED_EDGE_WEIGHT  * edge["weight"]})
            if not check:
                logging.error("cant find LLM score edge")
        
        # end of Dynamic adjust edge weight by LLM
         
        # Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight

        # Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        # Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        # Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)
        ppr_end = time.time()

        self.ppr_time += (ppr_end - ppr_start)

        # Restore the original graph 
        for edge_idx, original_weight in modified_edges.items():
            self.graph.es[edge_idx]["weight"] = original_weight
        
        assert len(ppr_sorted_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [
                ast.literal_eval(fact_row_dict[id]['content'])
                for id in real_candidate_fact_ids
            ]
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=True, #NOTE
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
    
    
    def call_llm_score(self, query, neighbors_key_all_phrase: List[str], entity_all_phrase: List[str], top_k_facts):
        all_qa_messages = []
        seed_triple_list = []
        for f in top_k_facts:
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            seed_triple_list.append(f'("{subject_phrase}", "{predicate_phrase}", "{object_phrase}")')
        
        seed_triplet_str = "; ".join(seed_triple_list)
        batch_neighbor_keys = []
        for neighbors_key, seed_entity in zip(neighbors_key_all_phrase, entity_all_phrase):
            neighbor_entities_str = ""
            current_query_keys = []
            
            neighbor_rows = self.entity_embedding_store.get_rows(neighbors_key)
            
            # compute the linking fact between the current seed node and neighbor node
            seed_node_fact = self.ent_node_to_fact_ids.get(compute_mdhash_id(content=seed_entity, prefix="entity-"), set())
            seed_fact_rows_dict = self.fact_embedding_store.get_rows(seed_node_fact)    # dict{h: {"hash_id": h, "content": t}}
            seed_fact_dict = {fact_id: seed_fact_rows_dict[fact_id]['content'] for fact_id in seed_node_fact}
            for idx, neighbor_phrase_key in enumerate(neighbors_key, 1):
                current_query_keys.append(neighbor_phrase_key)
                link_fact = self.ent_node_to_fact_ids.get(neighbor_phrase_key, set()).intersection(seed_node_fact)
                if len(link_fact) > 0:
                    link_fact_str = "; ".join([seed_fact_dict[f_id] for f_id in link_fact])
                # Case of Synthetic edge. We do not include synthetic edge in Edge Weighting and be filtered when the function call. 
                # Add log to Check no synthetic edge in LLM scoring. 
                else:
                    link_fact_str = "Semantic Similarity Link"
                    logging.error(f"Seed ({seed_entity}) and ({neighbor_rows[neighbor_phrase_key]['content']}) synthetic link")
            
                if neighbor_phrase_key in self.node_abs_store:
                    node_abs = self.node_abs_store[neighbor_phrase_key]
                    entity = node_abs["entity"]
                    summary = node_abs["summary"]
                    entity_info = f'[{idx}] "{entity}" | LINKING FACT: {link_fact_str} | SUMMARY: {summary}\n'
                else:
                    fact_ids_set = self.ent_node_to_fact_ids.get(neighbor_phrase_key, set())
                    fact_rows_dict = self.fact_embedding_store.get_rows(fact_ids_set)    # dict{h: {"hash_id": h, "content": t}}
                    fact_rows = [fact_rows_dict[fact_id] for fact_id in fact_ids_set]
                    entity_triplet_str = ""
                    for fact_row in fact_rows:
                        triple = tuple(ast.literal_eval(fact_row["content"]))
                        entity_triplet_str += f"{triple[0]} {triple[1]} {triple[2]}. "
                    entity_info = f'[{idx}] "{neighbor_rows[neighbor_phrase_key]["content"]}" | LINKING FACT: {link_fact_str} | SUMMARY: {entity_triplet_str}\n'
                
                neighbor_entities_str += entity_info
            
            batch_neighbor_keys.append(current_query_keys)
            if self.prompt_template_manager.is_template_name_valid(name="fact_score_with_sum"):
                pass
            else:
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template for dynamic score. Using MUSIQUE's prompt template instead.")
            all_qa_messages.append(
                self.prompt_template_manager.render(
                    name="fact_score_with_sum", 
                    query=query, 
                    seed_entity=seed_entity, 
                    seed_triplet=seed_triplet_str, 
                    neighbor_entities=neighbor_entities_str
                )
            )

        all_response_message = []
        all_metadata = []
    
        # batch_results = self.score_llm_model.batch_infer(all_qa_messages, max_completion_tokens=9192)
        batch_results = self.score_llm_model.batch_infer_async(all_qa_messages, max_completion_tokens=9192)

        # Process results
        for i, (response_message, metadata, cache_hit) in enumerate(batch_results):
            all_response_message.append(response_message)
            all_metadata.append(metadata)
        
        # 4. Process Results (ID Match -> Phrase Match Fallback)
        sort_neighbor_scores_all_phrase = []
        for response_text, original_keys in zip(all_response_message, batch_neighbor_keys):
            try:
                if "Answer:" in response_text:
                    response_text = response_text.split("Answer:", 1)[1].strip()
            except Exception as e:
                logger.warning(f"Error in split 'Answer' from raw LLM QA inference response: {str(e)}!")

            id_score_map = {}
            phrase_score_map = {}

            try:
                # 1: Extract ID-based scores
                # Matches: "1 (Name): 8" -> Captures '1' and '8'
                id_matches = re.findall(r"^(\d+)\s*\(.*?\)\s*:\s*(\d+)", response_text, re.MULTILINE)
                for m in id_matches:
                    id_score_map[int(m[0])] = int(m[1])

                # 2: Extract Phrase-based scores (Fallback)
                # Matches: "(Name): 8" -> Captures 'Name' and '8' inside parentheses
                # We use this if the ID lookup fails
                phrase_matches = re.findall(r"\((.*?)\)\s*:\s*(\d+)", response_text)
                for m in phrase_matches:
                    clean_name = text_processing(m[0])
                    phrase_score_map[clean_name] = int(m[1])
                    
            except Exception as e:
                logger.warning(f"Error parsing LLM response: {e}.")
                if response_text:
                    logger.warning(f" Raw response snippet: {response_text[:100]}...")


            # Reconstruct result list
            sort_neighbor_scores = []
            neighbor_rows = self.entity_embedding_store.get_rows(original_keys)
            
            for idx, key in enumerate(original_keys, 1):
                entity_name = neighbor_rows[key]['content']
                clean_entity_name = text_processing(entity_name)
                
                final_score = 4 # Default
                found = False
                
                # 1: Check ID
                if idx in id_score_map:
                    final_score = id_score_map[idx]
                    found = True
                # 2: Check Phrase Match
                elif clean_entity_name in phrase_score_map:
                    final_score = phrase_score_map[clean_entity_name]
                    found = True
                # 3: Check Phrase Match (Raw text containment - looser fallback)
                else:
                    for stored_name, score in phrase_score_map.items():
                        if clean_entity_name in stored_name or stored_name in clean_entity_name:
                            final_score = score
                            found = True
                            break
                
                if found == False:
                    logging.warning(f"`{entity_name}` not in sort_neighbor_scores. Set score as 4")

                
                sort_neighbor_scores.append({
                    "entity": entity_name, 
                    "score": final_score
                })

            sort_neighbor_scores_all_phrase.append(sort_neighbor_scores)
                
        return sort_neighbor_scores_all_phrase, all_metadata
