# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DPCoordinator - Data Parallel Coordinator for vLLM-Omni.

This module provides a coordinator for managing data parallel stage instances,
aggregating their status, and publishing updates for load balancing.

Example usage:

    from vllm_omni.distributed.dp_coordinator import (
        DPCoordinator, DPCoordinatorConfig,
        ClientForStage, ClientForStageConfig,
        ClientForHub, ClientForHubConfig,
        StageStatus,
    )

    # Start coordinator
    coord_config = DPCoordinatorConfig(
        pub_address="tcp://*:5555",
        router_address="tcp://*:5556",
    )
    coord = DPCoordinator(coord_config)
    coord.start()

    # Connect stage side client (worker side)
    stage_config = ClientForStageConfig(
        coordinator_addr="tcp://localhost:5556",
        zmq_addr="tcp://worker-1:8000",
    )
    stage_client = ClientForStage(stage_id=0, config=stage_config)
    stage_client.start()
    stage_client.set_status(StageStatus.UP)

    # Connect hub side client (API server side)
    hub_config = ClientForHubConfig(
        coordinator_addr="tcp://localhost:5555",
    )
    hub_client = ClientForHub(hub_config)
    hub_client.start()

    # Query available instances for load balancing
    ready_instances = hub_client.get_ready_instances(stage_id=0)
    least_loaded = hub_client.get_least_loaded_instance(stage_id=0)

    # Cleanup
    hub_client.stop()
    stage_client.stop()
    coord.stop()
"""

from .dp_coordinator import DPCoordinator, DPCoordinatorConfig
from .client_for_hub import ClientForHub, ClientForHubConfig
from .client_for_stage import ClientForStage, ClientForStageConfig
from .messages import (
    EventType,
    InstanceInfo,
    InstanceListing,
    StageStatus,
)

__all__ = [
    # Coordinator
    "DPCoordinator",
    "DPCoordinatorConfig",
    # Hub Side Client (API server side)
    "ClientForHub",
    "ClientForHubConfig",
    # Stage Side Client (worker side)
    "ClientForStage",
    "ClientForStageConfig",
    # Messages
    "EventType",
    "InstanceInfo",
    "InstanceListing",
    "StageStatus",
]
