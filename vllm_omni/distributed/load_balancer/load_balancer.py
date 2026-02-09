# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LoadBalancer - Task routing for vLLM-Omni distributed stages.

This module implements load balancers that route tasks to stage instances
based on different strategies.
"""

import hashlib
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

from vllm_omni.distributed.dp_coordinator.messages import InstanceInfo


@dataclass
class Task:
    """Placeholder for task object."""

    session_id: str
    request_id: str


class LoadBalancer(ABC):
    """Abstract base class for load balancers."""

    @abstractmethod
    def select(
        self,
        task: Task,
        instances: list[InstanceInfo],
    ) -> int:
        """Route a task to an instance.

        Args:
            task: The task to route
            instances: List of available instances

        Returns:
            Index of the selected instance
        """
        pass


class RandomBalancer(LoadBalancer):
    """Randomly select an instance from the available pool."""

    def select(
        self,
        task: Task,
        instances: list[InstanceInfo],
    ) -> int:
        if not instances:
            raise ValueError("No instances available")
        return random.randrange(len(instances))
