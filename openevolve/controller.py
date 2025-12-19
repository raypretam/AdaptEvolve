"""
Main controller for OpenEvolve
"""

import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import traceback
import numpy as np

from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    extract_code_language,
    extract_diffs,
    format_diff_summary,
    parse_evolve_blocks,
    parse_full_rewrite,
)
from openevolve.utils.format_utils import (
    format_metrics_safe,
    format_improvement_safe,
)

logger = logging.getLogger(__name__)


def _format_metrics(metrics: Dict[str, Any]) -> str:
    """Safely format metrics, handling both numeric and string values"""
    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={value}")
        else:
            formatted_parts.append(f"{name}={value}")
    return ", ".join(formatted_parts)


def _format_improvement(improvement: Dict[str, Any]) -> str:
    """Safely format improvement metrics"""
    formatted_parts = []
    for name, diff in improvement.items():
        if isinstance(diff, (int, float)) and not isinstance(diff, bool):
            try:
                formatted_parts.append(f"{name}={diff:+.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={diff}")
        else:
            formatted_parts.append(f"{name}={diff}")
    return ", ".join(formatted_parts)


class OpenEvolve:
    """
    Main controller for OpenEvolve

    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.

    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
        test_file_path: Optional[str] = None,
    ):
        # Load configuration
        if config is not None:
            # Use provided Config object directly
            self.config = config
        else:
            # Load from file or use defaults
            self.config = load_config(config_path)

        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "openevolve_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        self._setup_logging()
        self.test_file_path = test_file_path

        # Set random seed for reproducibility if specified
        if self.config.random_seed is not None:
            import random
            import numpy as np
            import hashlib

            # Set global random seeds
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            
            # Create hash-based seeds for different components
            base_seed = str(self.config.random_seed).encode('utf-8')
            llm_seed = int(hashlib.md5(base_seed + b'llm').hexdigest()[:8], 16) % (2**31)
            
            # Propagate seed to LLM configurations
            self.config.llm.random_seed = llm_seed
            for model_cfg in self.config.llm.models:
                if not hasattr(model_cfg, 'random_seed') or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed
            for model_cfg in self.config.llm.evaluator_models:
                if not hasattr(model_cfg, 'random_seed') or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed
            
            logger.info(f"Set random seed to {self.config.random_seed} for reproducibility")
            logger.debug(f"Generated LLM seed: {llm_seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        self.language = extract_code_language(self.initial_program_code)

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.config.llm.models, sampling=self.config.llm.sampling)
        self.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models, sampling=self.config.llm.sampling)

        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler.set_templates("evaluator_system_message")

        # Pass random seed to database if specified
        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed

        self.database = ProgramDatabase(self.config.database)

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=self.database,
        )

        logger.info(f"Initialized OpenEvolve with {initial_program_path} " f"and {evaluation_file}")

    def _setup_logging(self) -> None:
        """Set up logging"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        # Add file handler
        log_file = os.path.join(log_dir, f"openevolve_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
    ) -> Program:
        """
        Run the evolution process

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)

        Returns:
            Best program found
        """
        max_iterations = iterations or self.config.max_iterations

        # Define start_iteration before creating the initial program
        start_iteration = self.database.last_iteration

        # Only add initial program if starting fresh (not resuming from checkpoint)
        # Check if we're resuming AND no program matches initial code to avoid pollution
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            # Evaluate the initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id, self.test_file_path
            )

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                language=self.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
            )

            self.database.add(initial_program)
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} with {len(self.database.programs)} existing programs)"
            )

        # Main evolution loop
        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting evolution from iteration {start_iteration} for {max_iterations} iterations (total: {total_iterations})"
        )

        # Island-based evolution variables
        programs_per_island = max(
            1, max_iterations // (self.config.database.num_islands * 10)
        )  # Dynamic allocation
        current_island_counter = 0

        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()

        for i in range(start_iteration, total_iterations):
            iteration_start = time.time()

            # Manage island evolution - switch islands periodically
            if i > start_iteration and current_island_counter >= programs_per_island:
                self.database.next_island()
                current_island_counter = 0
                logger.debug(f"Switched to island {self.database.current_island}")

            current_island_counter += 1

            # Sample parent and inspirations from current island
            parent, inspirations = self.database.sample()

            # Get artifacts for the parent program if available
            parent_artifacts = self.database.get_artifacts(parent.id)

            # Get actual top programs for prompt context (separate from inspirations)
            # This ensures the LLM sees only high-performing programs as examples
            actual_top_programs = self.database.get_top_programs(1)

            # Build prompt
            prompt = self.prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,  # We don't have the parent's code, use the same
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in self.database.get_top_programs(1)],
                top_programs=[p.to_dict() for p in actual_top_programs],  # Use actual top programs
                inspirations=[p.to_dict() for p in inspirations],  # Pass inspirations separately
                language=self.language,
                evolution_round=i,
                diff_based_evolution=self.config.diff_based_evolution,
                program_artifacts=parent_artifacts if parent_artifacts else None,
            )

            # Generate code modification
            try:
                # generate N children from the LLM
                # import secrets

                N = 1
                responses: List[Tuple[str,int]] = await self.llm_ensemble.generate_multiple_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                    n=N,
                )

                # parse each LLM response into (code, uuid)
                children: List[Tuple[str, str, str, str, dict, dict, str]] = []  # (response, code, id, model_idx, confidence, verbalized_conf, model)
                for choice, model_idx in responses:
                    llm_response = choice.message.content
                    confidence = getattr(choice, "confidence", {"mean_confidence": 0})
                    verbalized_conf = getattr(choice, "verbalized_conf", {"verbalized_conf": 0})
                    model_called = getattr(choice, "llm", {"llm" : "qwen3"})
                    child_response = llm_response
                    print('=RESPONSE='*20)
                    print(llm_response)
                    # Parse the response
                    if self.config.diff_based_evolution:
                        diff_blocks = extract_diffs(llm_response)

                        if not diff_blocks:
                            logger.warning(f"Iteration {i+1}: No valid diffs found in response")
                            continue

                        # Apply the diffs
                        child_code = apply_diff(parent.code, llm_response)
                        changes_summary = format_diff_summary(diff_blocks)
                    else:
                        # Parse full rewrite
                        new_code = parse_full_rewrite(llm_response, self.language)
                        # code_from_think_response = parse_evolve_blocks(code)
                        # new_code = code_from_think_response
                        print(f'=CODE='*20)
                        print(new_code)

                        if not new_code:
                            logger.warning(f"Iteration {i+1}: No valid code found in response")
                            continue

                        child_code = new_code
                        changes_summary = "Full rewrite"

                    # Check code length
                    if len(child_code) > self.config.max_code_length:
                        logger.warning(
                            f"Iteration {i+1}: Generated code exceeds maximum length "
                            f"({len(child_code)} > {self.config.max_code_length})"
                        )
                        continue
                    
                    child_id = str(uuid.uuid4())
                    children.append((child_response, child_code, child_id, model_idx, confidence, verbalized_conf, model_called))

                if not children:
                    logger.warning(f"Iteration {i+1}: no valid children generated, skipping")
                    continue

                # Evaluate the child program
                # child_id = str(uuid.uuid4())
                # child_metrics = await self.evaluator.evaluate_multiple(child_code, child_id)
                # 3) Batch‐evaluate all children in parallel

                # Evaluator.evaluate_multiple wants List[Tuple[code, program_id]]
                eval_inputs = [(code, pid) for _, code, pid, _, _, _, _ in children]
                metrics_list = await self.evaluator.evaluate_multiple(eval_inputs, self.test_file_path)

                for child_no, ((child_response, child_code, child_id, model_idx, child_confidence, child_verbalized_conf, child_model), child_metrics) in enumerate(zip(children, metrics_list)):
                    # Handle artifacts if they exist
                    artifacts = self.evaluator.get_pending_artifacts(child_id)
                    # Create a child program
                    child_program = Program(
                        id=child_id,
                        code=child_code,
                        language=self.language,
                        parent_id=parent.id,
                        generation=parent.generation + 1,
                        metrics=child_metrics,
                        iteration_found=i + 1,
                        metadata={
                            "changes": changes_summary,
                            "parent_metrics": parent.metrics,
                            "confidence":  child_confidence,
                            "response": child_response,
                            "child_number": child_no,
                            "model": child_model,
                        },
                    )

                    # Update sampling function how this arm did
                    reward = self._calculate_reward(parent, child_metrics)
                    print(f'Reward : {reward}')
                    # # self.llm_ensemble.update_sampling_model(model_idx, reward)

                    # Add to database (will be added to current island)
                    self.database.add(child_program, iteration=i + 1)

                    # Log prompts
                    self.database.log_prompt(
                        template_key=(
                            "full_rewrite_user" if not self.config.diff_based_evolution else "diff_user"
                        ),
                        program_id=child_id,
                        prompt=prompt,
                        responses=[llm_response],
                    )

                    # Store artifacts if they exist
                    if artifacts:
                        self.database.store_artifacts(child_id, artifacts)

                    # Log prompts
                    self.database.log_prompt(
                        template_key=(
                            "full_rewrite_user" if not self.config.diff_based_evolution else "diff_user"
                        ),
                        program_id=child_id,
                        prompt=prompt,
                        responses=[llm_response],
                    )

                    # Increment generation for current island
                    self.database.increment_island_generation()

                    # Check if migration should occur
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {i+1}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    # Log progress
                    iteration_time = time.time() - iteration_start
                    self._log_iteration(i, parent, child_program, child_no, iteration_time)

                    # Specifically check if this is the new best program
                    if self.database.best_program_id == child_program.id:
                        logger.info(f"🌟 New best solution found at iteration {i+1}: {child_program.id}")
                        logger.info(f"Metrics: {format_metrics_safe(child_program.metrics)}")

                    # Save checkpoint
                    if (i + 1) % self.config.checkpoint_interval == 0:
                        self._save_checkpoint(i + 1)
                        # Also log island status at checkpoints
                        logger.info(f"Island status at checkpoint {i+1}:")
                        self.database.log_island_status()

                    # Check if target score reached
                    if target_score is not None:
                        # Only consider numeric metrics for target score calculation
                        numeric_metrics = [
                            v
                            for v in child_metrics.values()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        ]
                        reward  = (sum(numeric_metrics)/len(numeric_metrics)) if numeric_metrics else 0.0

                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(
                                    f"Target score {target_score} reached after {i+1} iterations"
                                )
                                break

            except Exception as e:
                logger.exception(f"Error in iteration {i+1}: {str(e)}")
                continue
            # exiting if correct code has been generated
            if reward == 1.0:
                logger.info(f"All Test Cases passed at Iteration: {i}")
                break
        # Get the best program using our tracking mechanism
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        # Fallback to calculating best program if tracked program not found
        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        # Check if there's a better program by combined_score that wasn't tracked
        if "combined_score" in best_program.metrics:
            best_by_combined = self.database.get_best_program(metric="combined_score")
            if (
                best_by_combined
                and best_by_combined.id != best_program.id
                and "combined_score" in best_by_combined.metrics
            ):
                # If the combined_score of this program is significantly better, use it instead
                if (
                    best_by_combined.metrics["combined_score"]
                    > best_program.metrics["combined_score"] + 0.02
                ):
                    logger.warning(
                        f"Found program with better combined_score: {best_by_combined.id}"
                    )
                    logger.warning(
                        f"Score difference: {best_program.metrics['combined_score']:.4f} vs {best_by_combined.metrics['combined_score']:.4f}"
                    )
                    best_program = best_by_combined

        if best_program:
            logger.info(
                f"Evolution complete. Best program has metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

            # Save the best program (using our tracked best program)
            self._save_best_program(best_program)

            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            # Return None if no programs found instead of undefined initial_program
            return None

    # def _log_iteration(
    #     self,
    #     iteration: int,
    #     parent: Program,
    #     child: Program,
    #     elapsed_time: float,
    # ) -> None:
    #     """
    #     Log iteration progress

    #     Args:
    #         iteration: Iteration number
    #         parent: Parent program
    #         child: Child program
    #         elapsed_time: Elapsed time in seconds
    #     """
    #     # Calculate improvement using safe formatting
    #     improvement_str = format_improvement_safe(parent.metrics, child.metrics)

    #     logger.info(
    #         f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
    #         f"in {elapsed_time:.2f}s. Metrics: "
    #         f"{format_metrics_safe(child.metrics)} "
    #         f"(Δ: {improvement_str})"
    #     )

    def _calculate_reward(self, parent: Program, child_metrics: dict) -> float:
        """Calculate reward for the asymmetric sampler"""
        metric_name = 'combined_score'  # or whatever your primary metric is
        
        parent_score = parent.metrics.get(metric_name, 0.0)
        child_score = child_metrics.get(metric_name, 0.0)
        reward = min(child_score, 1.0)
        
        return reward
    
    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        child_no: int,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        # Calculate improvement using safe formatting
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child Number :{child_no} - Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create specific checkpoint directory
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the database
        self.database.save(checkpoint_path, iteration)

        # Save the best program found so far
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
        else:
            best_program = self.database.get_best_program()

        if best_program:
            # Save the best program at this checkpoint
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            # Save metrics
            best_program_info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(best_program_info_path, "w") as f:
                import json

                json.dump(
                    {
                        "id": best_program.id,
                        "generation": best_program.generation,
                        "iteration": best_program.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best_program.metrics,
                        "language": best_program.language,
                        "timestamp": best_program.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                )

            logger.info(
                f"Saved best program at checkpoint {iteration} with metrics: "
                f"{format_metrics_safe(best_program.metrics)}"
            )

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
