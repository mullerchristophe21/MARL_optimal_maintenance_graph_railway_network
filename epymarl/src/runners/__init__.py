REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .parallel_runner_sep_rew import ParallelRunner_SR
REGISTRY["parallel_SR"] = ParallelRunner_SR