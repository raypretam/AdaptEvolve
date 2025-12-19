import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from openevolve.llm.base import LLMInterface
from openevolve.llm.openai import OpenAILLM
from openevolve.config import LLMModelConfig
from openevolve.llm.sampling import get_sampling_function

logger = logging.getLogger(__name__)

class LLMEnsemble:
    """Ensemble of LLMs with pluggable sampling strategies."""

    def __init__(
        self,
        models_cfg: List[LLMModelConfig],
        sampling: Optional[Dict[str, Any]] = None
    ):
        sampling = sampling or {}
        self.models_cfg = models_cfg
        self.models     = [OpenAILLM(cfg) for cfg in models_cfg]
        self.n_models   = len(models_cfg)

        # normalize static weights for random sampling if needed
        self.weights = [mc.weight for mc in models_cfg]
        total = sum(self.weights) or 1.0
        self.weights = [w / total for w in self.weights]

        # pick sampling function & kwargs
        fn_name = sampling.get("fn", "random")
        sampler_kwargs = {k: v for k, v in sampling.items() if k != "fn"}

        # random sampler needs the weights
        if fn_name == "random":
            sampler_kwargs["weights"] = self.weights
        logger.info(f"LLMEnsemble: using '{fn_name}' sampler with {sampler_kwargs}")
        self.sampler = get_sampling_function(fn_name, n_models=self.n_models, **sampler_kwargs)

        # deterministic RNG for any other needs
        self.random_state = random.Random()
        if hasattr(models_cfg[0], "random_seed"):
            self.random_state.seed(models_cfg[0].random_seed)

    def _sample_model_index(self) -> int:
        idx = self.sampler.sample()
        logger.info(f"Sampler picked model idx={idx} ({self.models_cfg[idx].name})")
        return idx

    async def generate(self, prompt: str, **kwargs) -> str:
        idx   = self._sample_model_index()
        model = self.models[idx]
        return await model.generate(prompt, **kwargs)

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str,str]], **kwargs
    ) -> Tuple[str,int]:
        idx   = self._sample_model_index()
        model = self.models[idx]
        resp  = await model.generate_with_context(system_message, messages, **kwargs)
        return resp, idx

    def update_sampling_model(self, model_idx: int, reward: float):
        print(f" Inside sampling model update")
        if hasattr(self.sampler, "update"):
            # Update the sampler's internal state first
            self.sampler.update(model_idx, reward)
            logger.debug(f"Updated sampler arm={model_idx} with reward={reward}")

        else:
            logger.warning("Sampler has no update() method; skipping.")
        curr_belief_state = self.sampler.get_belief_state()
        logger.info(curr_belief_state)


    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def generate_multiple_with_context(
        self, system_message: str, messages: List[Dict[str,str]], n: int, **kwargs
    ) -> List[Tuple[str,int]]:
        tasks = [
            self.generate_with_context(system_message, messages, **kwargs)
            for _ in range(n)
        ]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        tasks = [self.generate(p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str,str]], **kwargs
    ) -> List[str]:
        results = []
        for model in self.models:
            results.append(await model.generate_with_context(system_message, messages, **kwargs))
        return results
