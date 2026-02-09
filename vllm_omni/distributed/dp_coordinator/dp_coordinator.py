# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DPCoordinator - Data Parallel Coordinator for vLLM-Omni.

This module implements the coordinator that aggregates stage instance status
and publishes updates via ZMQ PUB/SUB for load balancing.
"""

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import zmq
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

from .messages import (
    EventType,
    InstanceInfo,
    InstanceListing,
    StageStatus,
)

logger = init_logger(__name__)


@dataclass
class DPCoordinatorConfig:
    """Configuration for DPCoordinator.

    Attributes:
        pub_address: Address for PUB socket (broadcasts instance listings)
        router_address: Address for ROUTER socket (receives registrations/heartbeats)
        heartbeat_timeout_ms: Time before marking an instance as stale
        publish_interval_ms: Interval between instance listing broadcasts
    """

    pub_address: str = "tcp://*:5555"
    router_address: str = "tcp://*:5556"
    heartbeat_timeout_ms: int = 5000
    publish_interval_ms: int = 500


class DPCoordinator:
    """Data Parallel Coordinator for managing stage instances.

    The coordinator maintains a registry of stage instances, receives
    heartbeats and status updates, and broadcasts instance listings
    for load balancing.

    This class implements the singleton pattern - use get_instance() to
    obtain the coordinator instance.
    """

    _instance: "DPCoordinator | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self, config: DPCoordinatorConfig | None = None):
        """Initialize the coordinator.

        Args:
            config: Configuration for the coordinator. If None, uses defaults.
        """
        self._config = config or DPCoordinatorConfig()

        # ZMQ context and sockets
        self._zmq_context: zmq.Context | None = None
        self._pub_socket: zmq.Socket | None = None
        self._router_socket: zmq.Socket | None = None

        # Thread-safe registry
        self._registry: dict[str, InstanceInfo] = {}
        self._registry_lock = threading.RLock()

        # Background threads
        self._receiver_thread: threading.Thread | None = None
        self._publisher_thread: threading.Thread | None = None
        self._health_checker_thread: threading.Thread | None = None

        # Control flags
        self._running = False
        self._stop_event = threading.Event()

    @classmethod
    def get_instance(cls, config: DPCoordinatorConfig | None = None) -> "DPCoordinator":
        """Get or create the singleton coordinator instance.

        Args:
            config: Configuration for the coordinator (only used on first call)

        Returns:
            The singleton DPCoordinator instance
        """
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Primarily for testing."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.stop()
                cls._instance = None

    @property
    def config(self) -> DPCoordinatorConfig:
        """Get the coordinator configuration."""
        return self._config

    def start(self, zmq_context: zmq.Context | None = None) -> None:
        """Start the coordinator and all background threads."""
        if self._running:
            logger.warning("DPCoordinator is already running")
            return

        logger.info("Starting DPCoordinator...")

        # Initialize ZMQ
        if zmq_context is None:
            self._zmq_context = zmq.Context()
            zmq_context = self._zmq_context

        # PUB socket for broadcasting instance listings
        self._pub_socket = zmq_context.socket(zmq.PUB)
        self._pub_socket.bind(self._config.pub_address)
        logger.info(f"PUB socket bound to {self._config.pub_address}")

        # ROUTER socket for receiving registrations and heartbeats
        self._router_socket = zmq_context.socket(zmq.ROUTER)
        self._router_socket.bind(self._config.router_address)
        self._router_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout for polling
        logger.info(f"ROUTER socket bound to {self._config.router_address}")

        self._running = True
        self._stop_event.clear()

        # Start background threads
        self._receiver_thread = threading.Thread(
            target=self._receiver_loop, name="DPCoordinator-Receiver", daemon=True
        )
        self._publisher_thread = threading.Thread(
            target=self._publisher_loop, name="DPCoordinator-Publisher", daemon=True
        )
        self._health_checker_thread = threading.Thread(
            target=self._health_checker_loop,
            name="DPCoordinator-HealthChecker",
            daemon=True,
        )

        self._receiver_thread.start()
        self._publisher_thread.start()
        self._health_checker_thread.start()

        logger.info("DPCoordinator started successfully")

    def stop(self) -> None:
        """Stop the coordinator and all background threads."""
        if not self._running:
            return

        logger.info("Stopping DPCoordinator...")
        self._running = False
        self._stop_event.set()

        # Wait for threads to finish
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=2.0)
        if self._publisher_thread and self._publisher_thread.is_alive():
            self._publisher_thread.join(timeout=2.0)
        if self._health_checker_thread and self._health_checker_thread.is_alive():
            self._health_checker_thread.join(timeout=2.0)

        # Close sockets
        if self._pub_socket:
            self._pub_socket.close()
            self._pub_socket = None
        if self._router_socket:
            self._router_socket.close()
            self._router_socket = None
        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None

        logger.info("DPCoordinator stopped")

    def register_instance(
        self,
        instance_id: str,
        stage_id: int,
        status: StageStatus = StageStatus.DOWN,
        queue_length: int = 0,
    ) -> None:
        """Register a new stage instance.

        Args:
            instance_id: Unique identifier for this instance (zmq_addr)
            stage_id: Identifier for the stage type
            status: Initial status
            queue_length: Initial queue length
        """
        timestamp = time.time()
        instance_info = InstanceInfo(
            stage_id=stage_id,
            instance_id=instance_id,
            status=status,
            queue_length=queue_length,
            last_heartbeat=timestamp,
            registered_at=timestamp,
        )

        with self._registry_lock:
            if instance_id in self._registry:
                logger.warning(f"Instance {instance_id} already registered, updating")
            self._registry[instance_id] = instance_info

        logger.info(f"Registered instance {instance_id} (stage={stage_id})")

    def unregister_instance(self, instance_id: str) -> bool:
        """Unregister a stage instance.

        Args:
            instance_id: The instance ID (zmq_addr) to unregister

        Returns:
            True if the instance was found and removed, False otherwise
        """
        with self._registry_lock:
            if instance_id in self._registry:
                del self._registry[instance_id]
                logger.info(f"Unregistered instance {instance_id}")
                return True
            else:
                logger.warning(f"Instance {instance_id} not found for unregistration")
                return False

    def update_instance(
        self,
        instance_id: str,
        status: StageStatus | None = None,
        queue_length: int | None = None,
    ) -> bool:
        """Update the status and/or queue length of an instance.

        Args:
            instance_id: The instance ID (zmq_addr) to update
            status: New status (optional)
            queue_length: New queue length (optional)

        Returns:
            True if the instance was found and updated, False otherwise
        """
        timestamp = time.time()

        with self._registry_lock:
            if instance_id not in self._registry:
                return False

            instance = self._registry[instance_id]
            instance.last_heartbeat = timestamp

            if status is not None:
                instance.status = status
            if queue_length is not None:
                instance.queue_length = queue_length

        return True

    def get_instance_listing(self) -> InstanceListing:
        """Get the current instance listing.

        Returns:
            InstanceListing containing all registered instances
        """
        with self._registry_lock:
            instances = list(self._registry.values())

        return InstanceListing(instances=instances, timestamp=time.time())

    def get_instances_by_stage(self, stage_id: int) -> list[InstanceInfo]:
        """Get all instances for a specific stage.

        Args:
            stage_id: The stage ID to filter by

        Returns:
            List of InstanceInfo for the specified stage
        """
        with self._registry_lock:
            return [
                inst for inst in self._registry.values() if inst.stage_id == stage_id
            ]

    def get_instance(self, instance_id: str) -> InstanceInfo | None:
        """Get a specific instance by ID.

        Args:
            instance_id: The instance ID (zmq_addr) to look up
        Returns:
            InstanceInfo if found, None otherwise
        """
        with self._registry_lock:
            return self._registry.get(instance_id)

    def health(self) -> dict[str, Any]:
        """Get coordinator health status.

        Returns:
            Dictionary with health information
        """
        with self._registry_lock:
            total_instances = len(self._registry)
            ready_instances = sum(
                1
                for inst in self._registry.values()
                if inst.status == StageStatus.UP
            )
            instances_by_stage: dict[int, int] = {}
            for inst in self._registry.values():
                instances_by_stage[inst.stage_id] = (
                    instances_by_stage.get(inst.stage_id, 0) + 1
                )

        return {
            "running": self._running,
            "total_instances": total_instances,
            "ready_instances": ready_instances,
            "instances_by_stage": instances_by_stage,
            "pub_address": self._config.pub_address,
            "router_address": self._config.router_address,
        }

    def _receiver_loop(self) -> None:
        """Background thread that receives messages from clients."""
        logger.debug("Receiver thread started")

        while self._running and not self._stop_event.is_set():
            try:
                # ROUTER socket receives: [identity, empty, message]
                frames = self._router_socket.recv_multipart()
                if len(frames) < 2:
                    logger.warning(f"Received malformed message: {len(frames)} frames")
                    continue

                identity = frames[0]
                message_data = frames[-1]  # Last frame is the message

                self._handle_message(identity, message_data)

            except zmq.Again:
                # Timeout, continue loop
                continue
            except zmq.ZMQError as e:
                if self._running:
                    logger.error(f"ZMQ error in receiver: {e}")
            except Exception as e:
                if self._running:
                    logger.error(f"Error in receiver loop: {e}")

        logger.debug("Receiver thread stopped")

    def _handle_message(self, identity: bytes, message_data: bytes) -> None:
        """Handle an incoming message from a client.

        Args:
            identity: ZMQ identity of the sender (zmq_addr)
            message_data: Serialized message data
        """
        try:
            message = OmniSerializer.deserialize(message_data)

            if not isinstance(message, dict):
                logger.warning(f"Received non-dict message: {type(message)}")
                return

            event_type = message.get("event_type")
            if event_type is None:
                logger.warning("Received message without event_type")
                return

            # Instance ID is the ZMQ identity (zmq_addr)
            instance_id = identity.decode("utf-8")

            # Extract common fields from message
            stage_id = message.get("stage_id", 0)
            status_str = message.get("status")
            status = StageStatus(status_str) if status_str else None
            queue_length = message.get("queue_length", 0)

            if event_type == EventType.STATUS_UPDATE or event_type == "status_update":
                self._handle_status_update(instance_id, stage_id, status, queue_length)
            elif event_type == EventType.LOAD_UPDATE or event_type == "load_update":
                self._handle_load_update(instance_id, stage_id, status, queue_length)
            elif event_type == EventType.HEARTBEAT or event_type == "heartbeat":
                self._handle_heartbeat(instance_id, stage_id, status, queue_length)
            else:
                logger.warning(f"Unknown event type: {event_type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    def _handle_status_update(
        self,
        instance_id: str,
        stage_id: int,
        status: StageStatus | None,
        queue_length: int,
    ) -> None:
        """Handle a status update message.

        Auto-registers unknown instances and unregisters instances with DOWN status.
        """
        if status == StageStatus.DOWN:
            self.unregister_instance(instance_id)
            return

        # Auto-register if instance is unknown
        if self.get_instance(instance_id) is None:
            self.register_instance(
                instance_id=instance_id,
                stage_id=stage_id,
                status=status or StageStatus.DOWN,
                queue_length=queue_length,
            )
            if status == StageStatus.UP:
                logger.info(f"Instance {instance_id} is now UP")
        elif status:
            self.update_instance(instance_id, status=status, queue_length=queue_length)
            if status == StageStatus.UP:
                logger.info(f"Instance {instance_id} is now UP")
            elif status == StageStatus.ERROR:
                logger.error(f"Instance {instance_id} reported error")

    def _handle_heartbeat(
        self,
        instance_id: str,
        stage_id: int,
        status: StageStatus | None,
        queue_length: int,
    ) -> None:
        """Handle a heartbeat message."""
        # Auto-register if instance is unknown
        if self.get_instance(instance_id) is None:
            self.register_instance(
                instance_id=instance_id,
                stage_id=stage_id,
                status=status or StageStatus.DOWN,
                queue_length=queue_length,
            )
        else:
            self.update_instance(instance_id, status=status, queue_length=queue_length)

    def _handle_load_update(
        self,
        instance_id: str,
        stage_id: int,
        status: StageStatus | None,
        queue_length: int,
    ) -> None:
        """Handle a load update message."""
        # Auto-register if instance is unknown
        if self.get_instance(instance_id) is None:
            self.register_instance(
                instance_id=instance_id,
                stage_id=stage_id,
                status=status or StageStatus.DOWN,
                queue_length=queue_length,
            )
        else:
            self.update_instance(instance_id, status=status, queue_length=queue_length)

    def _publisher_loop(self) -> None:
        """Background thread that publishes instance listings periodically."""
        logger.debug("Publisher thread started")
        publish_interval = self._config.publish_interval_ms / 1000.0

        while self._running and not self._stop_event.is_set():
            try:
                listing = self.get_instance_listing()
                self._publish_listing(listing)
            except Exception as e:
                if self._running:
                    logger.error(f"Error in publisher loop: {e}")

            self._stop_event.wait(publish_interval)

        logger.debug("Publisher thread stopped")

    def _publish_listing(self, listing: InstanceListing) -> None:
        """Publish an instance listing to subscribers.

        Args:
            listing: The instance listing to publish
        """
        if not self._pub_socket:
            return

        try:
            # Convert to serializable dict
            listing_dict = {
                "instances": [asdict(inst) for inst in listing.instances],
                "timestamp": listing.timestamp,
            }
            # Convert enum values to strings
            for inst in listing_dict["instances"]:
                if isinstance(inst.get("status"), StageStatus):
                    inst["status"] = inst["status"].value
                elif inst.get("status"):
                    inst["status"] = str(inst["status"])

            data = OmniSerializer.serialize(listing_dict)
            self._pub_socket.send_multipart([b"listing", data])

        except Exception as e:
            logger.error(f"Error publishing listing: {e}")

    def _health_checker_loop(self) -> None:
        """Background thread that detects stale instances."""
        logger.debug("Health checker thread started")
        check_interval = self._config.heartbeat_timeout_ms / 1000.0 / 2.0

        while self._running and not self._stop_event.is_set():
            try:
                self._check_stale_instances()
            except Exception as e:
                if self._running:
                    logger.error(f"Error in health checker loop: {e}")

            self._stop_event.wait(check_interval)

        logger.debug("Health checker thread stopped")

    def _check_stale_instances(self) -> None:
        """Check for and handle stale instances."""
        current_time = time.time()
        timeout_seconds = self._config.heartbeat_timeout_ms / 1000.0
        stale_instances: list[str] = []

        with self._registry_lock:
            for instance_id, instance in self._registry.items():
                if instance.status == StageStatus.DOWN:
                    continue
                if current_time - instance.last_heartbeat > timeout_seconds:
                    stale_instances.append(instance_id)
        for instance_id in stale_instances:
            logger.warning(f"Instance {instance_id} is stale (no heartbeat)")
            self.update_instance(instance_id, status=StageStatus.ERROR)


def main():
    """Entry point for running the coordinator as a standalone process."""
    import argparse

    parser = argparse.ArgumentParser(description="DPCoordinator - Data Parallel Coordinator")
    parser.add_argument(
        "--pub-address",
        type=str,
        default="tcp://*:5555",
        help="PUB socket bind address (default: tcp://*:5555)",
    )
    parser.add_argument(
        "--router-address",
        type=str,
        default="tcp://*:5556",
        help="ROUTER socket bind address (default: tcp://*:5556)",
    )
    parser.add_argument(
        "--heartbeat-timeout-ms",
        type=int,
        default=5000,
        help="Heartbeat timeout in milliseconds (default: 5000)",
    )
    parser.add_argument(
        "--publish-interval-ms",
        type=int,
        default=500,
        help="Instance listing publish interval in milliseconds (default: 500)",
    )

    args = parser.parse_args()

    config = DPCoordinatorConfig(
        pub_address=args.pub_address,
        router_address=args.router_address,
        heartbeat_timeout_ms=args.heartbeat_timeout_ms,
        publish_interval_ms=args.publish_interval_ms,
    )

    coordinator = DPCoordinator(config)
    coordinator.start()

    print(f"DPCoordinator started")
    print(f"  PUB address: {config.pub_address}")
    print(f"  ROUTER address: {config.router_address}")
    print("Press Ctrl+C to stop...")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        coordinator.stop()


if __name__ == "__main__":
    main()
