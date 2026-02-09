# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ClientForHub - Client for API servers to communicate with DPCoordinator.

This module implements the client that API servers (hubs) use to receive
instance listings from the coordinator for load balancing purposes.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import zmq
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

from .messages import InstanceInfo, InstanceListing, StageStatus

logger = init_logger(__name__)


@dataclass
class ClientForHubConfig:
    """Configuration for ClientForHub.

    Attributes:
        coordinator_addr: Address of the coordinator PUB socket
        reconnect_interval_ms: Interval between reconnection attempts
        stale_threshold_ms: Time after which cached data is considered stale
    """

    coordinator_addr: str = "tcp://localhost:5555"
    reconnect_interval_ms: int = 5000
    stale_threshold_ms: int = 10000


class ClientForHub:
    """Client for API servers to receive instance listings from DPCoordinator.

    This client subscribes to instance listing broadcasts from the coordinator
    and maintains a local cache of available stage instances for load balancing.
    """

    def __init__(self, config: ClientForHubConfig | None = None):
        """Initialize the hub side client.

        Args:
            config: Client configuration (optional)
        """
        self._config = config or ClientForHubConfig()

        # ZMQ
        self._zmq_context: zmq.Context | None = None
        self._sub_socket: zmq.Socket | None = None

        # Instance cache
        self._instance_cache: dict[str, InstanceInfo] = {}
        self._cache_lock = threading.RLock()
        self._last_update: float = 0.0

        # Background receiver thread
        self._receiver_thread: threading.Thread | None = None
        self._running = False
        self._stop_event = threading.Event()

        # Callbacks
        self._on_update_callbacks: list[Callable[[InstanceListing], None]] = []

    @property
    def last_update(self) -> float:
        """Get the timestamp of the last update."""
        return self._last_update

    @property
    def is_stale(self) -> bool:
        """Check if the cached data is stale."""
        if self._last_update == 0.0:
            return True
        age_ms = (time.time() - self._last_update) * 1000
        return age_ms > self._config.stale_threshold_ms

    def start(self， zmq_context: zmq.Context | None = None) -> None:
        """Start the client and connect to the coordinator."""
        if self._running:
            logger.warning("ClientForHub is already running")
            return

        logger.info("Starting ClientForHub...")

        # Initialize ZMQ
        if zmq_context is None:
            self._zmq_context = zmq.Context()
            zmq_context = self._zmq_context

        # SUB socket for receiving instance listings
        self._sub_socket = zmq_context.socket(zmq.SUB)
        self._sub_socket.connect(self._config.coordinator_addr)
        self._sub_socket.setsockopt_string(zmq.SUBSCRIBE, "listing")
        self._sub_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout for polling
        logger.info(
            f"Subscribed to coordinator at {self._config.coordinator_addr}"
        )

        self._running = True
        self._stop_event.clear()

        # Start receiver thread
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop,
            name="ClientForHub-Receiver",
            daemon=True,
        )
        self._receiver_thread.start()

        logger.info("ClientForHub started")

    def stop(self) -> None:
        """Stop the client and disconnect from the coordinator."""
        if not self._running:
            return

        logger.info("Stopping ClientForHub...")

        self._running = False
        self._stop_event.set()

        # Wait for receiver thread
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2.0)

        # Close sockets
        if self._sub_socket:
            self._sub_socket.close()
            self._sub_socket = None
        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None

        logger.info("ClientForHub stopped")

    def add_update_callback(
        self, callback: Callable[[InstanceListing], None]
    ) -> None:
        """Add a callback to be called when instance listing is updated.

        Args:
            callback: Function to call with the new InstanceListing
        """
        self._on_update_callbacks.append(callback)

    def remove_update_callback(
        self, callback: Callable[[InstanceListing], None]
    ) -> None:
        """Remove a previously added callback.

        Args:
            callback: The callback function to remove
        """
        if callback in self._on_update_callbacks:
            self._on_update_callbacks.remove(callback)

    def get_instance_listing(self) -> InstanceListing:
        """Get the current instance listing from cache.

        Returns:
            InstanceListing containing all known instances
        """
        with self._cache_lock:
            instances = list(self._instance_cache.values())
            return InstanceListing(instances=instances, timestamp=self._last_update)

    def get_instances_by_stage(self, stage_id: int) -> list[InstanceInfo]:
        """Get all instances for a specific stage.

        Args:
            stage_id: The stage ID to filter by

        Returns:
            List of InstanceInfo for the specified stage
        """
        with self._cache_lock:
            return [
                inst
                for inst in self._instance_cache.values()
                if inst.stage_id == stage_id
            ]

    def get_ready_instances(self, stage_id: int | None = None) -> list[InstanceInfo]:
        """Get all instances that are ready to accept requests.

        Args:
            stage_id: Optional stage ID to filter by

        Returns:
            List of ready InstanceInfo,optionally filtered by stage
        """
        with self._cache_lock:
            instances = [
                inst
                for inst in self._instance_cache.values()
                if inst.status == StageStatus.UP
            ]
            if stage_id is not None:
                instances = [inst for inst in instances if inst.stage_id == stage_id]
            return instances

    def get_instance(self, zmq_addr: str) -> InstanceInfo | None:
        """Get a specific instance by ZMQ address.

        Args:
            zmq_addr: The ZMQ address of the instance to look up

        Returns:
            InstanceInfo if found, None otherwise
        """
        with self._cache_lock:
            return self._instance_cache.get(zmq_addr)

    def get_least_loaded_instance(
        self, stage_id: int | None = None
    ) -> InstanceInfo | None:
        """Get the instance with the lowest load.

        Args:
            stage_id: Optional stage ID to filter by

        Returns:
            InstanceInfo with lowest load, or None if no ready instances
        """
        instances = self.get_ready_instances(stage_id)
        if not instances:
            return None

        # Sort by queue length (number of unfinished tasks)
        return min(instances, key=lambda inst: inst.queue_length)

    def health(self) -> dict[str, Any]:
        """Get client health status.

        Returns:
            Dictionary with health information
        """
        with self._cache_lock:
            total_instances = len(self._instance_cache)
            ready_instances = sum(
                1
                for inst in self._instance_cache.values()
                if inst.status == StageStatus.UP
            )
            instances_by_stage: dict[int, int] = {}
            for inst in self._instance_cache.values():
                instances_by_stage[inst.stage_id] = (
                    instances_by_stage.get(inst.stage_id, 0) + 1
                )

        return {
            "running": self._running,
            "connected": self._running and not self.is_stale,
            "total_instances": total_instances,
            "ready_instances": ready_instances,
            "instances_by_stage": instances_by_stage,
            "last_update": self._last_update,
            "is_stale": self.is_stale,
            "coordinator_addr": self._config.coordinator_addr,
        }

    def _receiver_loop(self) -> None:
        """Background thread that receives instance listings."""
        logger.debug("Receiver thread started")

        while self._running and not self._stop_event.is_set():
            try:
                # Receive multipart message: [topic, data]
                frames = self._sub_socket.recv_multipart()
                if len(frames) >= 2:
                    topic = frames[0].decode("utf-8")
                    data = frames[1]

                    if topic == "listing":
                        self._handle_listing(data)

            except zmq.Again:
                # Timeout, continue loop
                continue
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ error in receiver: {e}")
                    self._stop_event.wait(
                        self._config.reconnect_interval_ms / 1000.0
                    )
            except Exception as e:
                if self._running:
                    logger.error(f"Error in receiver loop: {e}")

        logger.debug("Receiver thread stopped")

    def _handle_listing(self, data: bytes) -> None:
        """Handle an incoming instance listing.

        Args:
            data: Serialized listing data
        """
        try:
            listing_dict = OmniSerializer.deserialize(data)

            if not isinstance(listing_dict, dict):
                logger.warning(f"Received non-dict listing: {type(listing_dict)}")
                return

            # Parse instances
            instances: list[InstanceInfo] = []
            for inst_dict in listing_dict.get("instances", []):
                # Parse status
                status_str = inst_dict.get("status", "down")
                if isinstance(status_str, StageStatus):
                    status = status_str
                else:
                    try:
                        status = StageStatus(status_str)
                    except ValueError:
                        status = StageStatus.DOWN

                instance = InstanceInfo(
                    stage_id=inst_dict.get("stage_id", 0),
                    zmq_addr=inst_dict.get("zmq_addr", ""),
                    status=status,
                    queue_length=inst_dict.get("queue_length", 0),
                    last_heartbeat=inst_dict.get("last_heartbeat", 0.0),
                    registered_at=inst_dict.get("registered_at", 0.0),
                )
                instances.append(instance)

            # Update cache
            timestamp = listing_dict.get("timestamp", time.time())
            with self._cache_lock:
                self._instance_cache = {inst.zmq_addr: inst for inst in instances}
                self._last_update = timestamp

            # Notify callbacks
            listing = InstanceListing(instances=instances, timestamp=timestamp)
            for callback in self._on_update_callbacks:
                try:
                    callback(listing)
                except Exception as e:
                    logger.error(f"Error in update callback: {e}")

            logger.debug(f"Updated instance cache: {len(instances)} instances")

        except Exception as e:
            logger.error(f"Error handling listing: {e}")
