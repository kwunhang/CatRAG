import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig
import concurrent.futures
from openai import AsyncOpenAI, AsyncAzureOpenAI
import asyncio

logger = get_logger(__name__)

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if self.overhead_bypass_cache:
            # Directly call the function (no lock, no db, no overhead)
            result = func(self, *args, **kwargs)
            return result[0], result[1], False
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        n = kwargs.get("n", 1)

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        # for case if request multiple answer. Only add the n too key_data when n is not 1, to prevent hash_id is mismatch for cache
        if n != 1:
            key_data["n"] = n
            
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)  
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

# ASYNC CACHE
def async_cache_response(func):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Bypass Cache IF evaluating overhead
        if self.overhead_bypass_cache:
            # Directly call the function (no lock, no db, no overhead)
            result = await func(self, *args, **kwargs)
            return result[0], result[1], False
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        n = kwargs.get("n", 1)

        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        if n != 1:
            key_data["n"] = n
            
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        lock_file = self.cache_file_name + ".lock"

        # DB Operations
        def _read_from_cache_sync():
            with FileLock(lock_file):
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
                c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                row = c.fetchone()
                conn.close()
                return row

        def _write_to_cache_sync(msg, meta):
            with FileLock(lock_file):
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                """)
                meta_str = json.dumps(meta)
                c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                          (key_hash, msg, meta_str))
                conn.commit()
                conn.close()

        # Run Read in Thread Pool (Non-Blocking)
        loop = asyncio.get_running_loop()
        row = await loop.run_in_executor(None, _read_from_cache_sync)

        if row is not None:
            message, metadata_str = row
            metadata = json.loads(metadata_str)
            return message, metadata, True

        # Call Original Async Function (Cache Miss)
        result = await func(self, *args, **kwargs)
        message, metadata = result

        # Run Write in Thread Pool (Non-Blocking)
        await loop.run_in_executor(None, _write_to_cache_sync, message, metadata)

        return message, metadata, False

    return wrapper

# ASYNC RETRY DECORATOR
def async_dynamic_retry_decorator(func):
    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Await the decorated async function
                return await func(self, *args, **kwargs)
            except Exception as e:
                last_exception = e
                # Don't retry on fatal errors (optional)
                if "invalid_request_error" in str(e):
                    raise e
                    
                # logging.warning(f"Attempt {attempt+1} failed: {e}. Retrying...")
                if attempt < max_retries - 1:
                    # Async sleep (non-blocking)
                    await asyncio.sleep(1) 
        
        logger.error(f"Max retries reached. Last error: {last_exception}")
        raise last_exception
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True, global_meter=None,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url
        self.overhead_bypass_cache = global_config.overhead_bypass_cache
        self.global_meter = global_meter

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        self.openai_client = None
        if self.global_config.azure_endpoint is None:
            self.openai_client = OpenAI(base_url=self.llm_base_url, http_client=client, max_retries=self.max_retries)
            self.async_openai_client = AsyncOpenAI(base_url=self.llm_base_url, http_client=httpx.AsyncClient(), max_retries=self.max_retries)
        else:
            self.openai_client = AzureOpenAI(api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                                             azure_endpoint=self.global_config.azure_endpoint, max_retries=self.max_retries, http_client=httpx.Client())
            self.async_openai_client = AsyncAzureOpenAI(api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                                             azure_endpoint=self.global_config.azure_endpoint, max_retries=self.max_retries, http_client=httpx.AsyncClient())

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"):
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }
        if self.global_meter:
            self.global_meter.record(
                model_name=params.get("model", self.global_config.llm_name), 
                prompt_tokens=metadata["prompt_tokens"], 
                completion_tokens=metadata["completion_tokens"]
            )

        return response_message, metadata

    @cache_response
    @dynamic_retry_decorator
    def infer_multi_response(
        self,
        messages: List[TextChatMessage],
        n: int,
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        params["n"] = n
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"):
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(**params)

        response_messages = [choice.message.content for choice in response.choices]
        assert isinstance(response_messages[0], str), "response_message should be a string"
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return str(response_messages), metadata

    def batch_infer(self, messages_list, max_completion_tokens=9192, max_workers=8, **kwargs):
        """
        Process multiple LLM inference requests concurrently using the original infer function.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_index = {}
            for idx, messages in enumerate(messages_list):
                # Schedule the execution of the original LLM infer method
                future = executor.submit(
                    self.infer,  # The original LLM function
                    messages,    # The messages argument for infer()
                    max_completion_tokens=max_completion_tokens,  # The kwargs for infer()
                    **kwargs
                )
                future_to_index[future] = idx

            # Collect results, handling exceptions for each request
            results = [None] * len(messages_list)
            for future in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    response_message, metadata, cache = future.result()
                    results[idx] = (response_message, metadata, cache)
                except Exception as e:
                    # Handle exceptions for the individual request
                    if 'content_filter' in str(e):
                        logger.warning(f"Query {idx} was filtered by content policy. Error: {e}")
                        results[idx] = ("[ANSWER SKIPPED] Response was filtered by content policy.", {}, False)
                    else:
                        logger.error(f"Request {idx} failed: {str(e)}")
                        results[idx] = ("[REQUEST FAILED]", {}, False)
        return results
    
    @async_cache_response
    @async_dynamic_retry_decorator
    async def infer_async(self, messages, **kwargs):
        """
        Single Async Inference Step
        """
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        
        if 'gpt' not in params['model']:
            params['max_tokens'] = params.pop('max_completion_tokens', 4096)

        # Call Async API
        response = await self.async_openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }
        if self.global_meter:
            self.global_meter.record(
                model_name=params.get("model", self.global_config.llm_name), 
                prompt_tokens=metadata["prompt_tokens"], 
                completion_tokens=metadata["completion_tokens"]
            )
        
        return response_message, metadata

    async def batch_infer_async_driver(self, messages_list, max_completion_tokens=9192, concurrency_limit=20, **kwargs):
        """
        Manages the concurrent execution window (Semaphore).
        """
        sem = asyncio.Semaphore(concurrency_limit)

        async def llm_worker(idx, msg):
            async with sem:
                try:
                    # The decorators on infer_async handle the Cache and Retry logic automatically
                    msg_resp, meta, cache_hit = await self.infer_async(msg, max_completion_tokens=max_completion_tokens, **kwargs)
                    return (msg_resp, meta, cache_hit)
                except Exception as e:
                    logger.error(f"Request {idx} failed after retries: {e}")
                    return ("[REQUEST FAILED]", {}, False)

        tasks = [llm_worker(i, msg) for i, msg in enumerate(messages_list)]
        return await asyncio.gather(*tasks)

    def batch_infer_async(self, messages_list, **kwargs):
        """
        Entry point for synchronous code to start the async pipeline.
        """
        return asyncio.run(self.batch_infer_async_driver(messages_list, **kwargs))
