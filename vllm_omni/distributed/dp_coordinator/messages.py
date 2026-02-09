# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Message protocol definitions for DPCoordinator.

This module defines the dataclasses and enums used for communication
between the DPCoordinator and StageClients.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    """Status of a stage instance."""

    UP = "up"
    DOWN = "down"
    ERROR = "error"


class EventType(str, Enum):
    """Types of events sent between coordinator and clients."""

    STATUS_UPDATE = "status_update"
    LOAD_UPDATE = "load_update"
    HEARTBEAT = "heartbeat"


@dataclass
class InstanceInfo:
    """Instance state stored in the coordinator registry.

    Attributes:
        stage_id: Identifier for the stage type
        zmq_addr: Unique identifier for this instance
        status: Current status of the instance
        queue_length: Number of unfinished tasks
        last_heartbeat: Unix timestamp of last heartbeat
        registered_at: Unix timestamp when instance was registered
    """

    stage_id: int
    zmq_addr: str
    status: StageStatus = StageStatus.DOWN
    queue_length: int = 0
    last_heartbeat: float = 0.0
    registered_at: float = 0.0


@dataclass
class InstanceListing:
    """List of instances for PUB broadcast.

    Attributes:
        instances: List of instance information
        timestamp: Unix timestamp when listing was generated
    """

    instances: list[InstanceInfo] = field(default_factory=list)
    timestamp: float = 0.0

    def get_instances_by_stage(self, stage_id: int) -> list[InstanceInfo]:
        """Get all instances for a specific stage."""
        return [inst for inst in self.instances if inst.stage_id == stage_id]

    def get_ready_instances(self) -> list[InstanceInfo]:
        """Get all instances that are ready to accept requests."""
        return [inst for inst in self.instances if inst.status == StageStatus.UP]
