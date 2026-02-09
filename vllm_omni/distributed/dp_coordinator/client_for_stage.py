# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ClientForStage - Client for workers to communicate with DPCoordinator.

This module implements the client that stage workers use to register
with the coordinator, send heartbeats, and report status updates.
"""

import threading
import time
from dataclasses import dataclass

import zmq
from vllm.logger import init_logger

from vllm_omni.distributed.omni_connectors.utils.serialization import OmniSerializer

from .messages import EventType, StageStatus

logger = init_logger(__name__)


@dataclass
class ClientForStageConfig:
    """Configuration for ClientForStage.

    Attributes:
        coordinator_addr: Address of the coordinator ROUTER socket
        zmq_addr: ZMQ address where this stage receives tasks
        heartbeat_interval_ms: Interval between heartbeats
    """

    coordinator_addr: str = "tcp://localhost:5556"
    zmq_addr: str = ""
    heartbeat_interval_ms: int = 1000


class ClientForStage:
    """Client for stage workers to communicate with the DPCoordinator.

    This client handles heartbeats and status updates for a stage instance.
    """

    def __init__(self, stage_id: int, config: ClientForStageConfig):
        """Initialize the stage client.

        Args:
            stage_id: Identifier for the stage (e.g., 0 for LLM, 1 for diffusion)
            config: Client configuration
        """
        self._config = config
        self._stage_id = stage_id
        self._status = StageStatus.DOWN
        self._queue_length = 0
        self._lock = threading.RLock()

        # ZMQ
        self._zmq_context: zmq.Context | None = None
        self._dealer_socket: zmq.Socket | None = None

        # Background heartbeat thread
        self._heartbeat_thread: threading.Thread | None = None
        self._running = False
        self._stop_event = threading.Event()

    @property
    def zmq_addr(self) -> str:
        """Get the ZMQ address."""
        return self._config.zmq_addr

    @property
    def stage_id(self) -> int:
        """Get the stage ID."""
        return self._stage_id

    @property
    def status(self) -> StageStatus:
        """Get the current status."""
        return self._status

    def start(self, zmq_context: zmq.Context | None = None) -> None:
        """Start the client and connect to the coordinator."""
        if self._running:
            logger.warning("ClientForStage is already running")
            return

        logger.info(f"Starting ClientForStage {self._config.zmq_addr}...")

        # Initialize ZMQ
        if zmq_context is None:
            self._zmq_context = zmq.Context()
            zmq_context = self._zmq_context

        # DEALER socket for communication with coordinator
        self._dealer_socket = zmq_context.socket(zmq.DEALER)
        self._dealer_socket.setsockopt_string(zmq.IDENTITY, self._config.zmq_addr)
        self._dealer_socket.connect(self._config.coordinator_addr)
        logger.info(f"Connected to coordinator at {self._config.coordinator_addr}")

        self._running = True
        self._stop_event.clear()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"ClientForStage-Heartbeat-{self._config.zmq_addr}",
            daemon=True,
        )
        self._heartbeat_thread.start()

        logger.info(f"ClientForStage {self._config.zmq_addr} started")

    def stop(self) -> None:
        """Stop the client and disconnect from the coordinator."""
        if not self._running:
            return

        logger.info(f"Stopping ClientForStage {self._config.zmq_addr}...")

        # Send DOWN status
        try:
            self.set_status(StageStatus.DOWN)
        except Exception as e:
            logger.warning(f"Failed to send status update: {e}")

        self._running = False
        self._stop_event.set()

        # Wait for heartbeat thread
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)

        # Close sockets
        if self._dealer_socket:
            self._dealer_socket.close()
            self._dealer_socket = None
        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None

        logger.info(f"ClientForStage {self._config.zmq_addr} stopped")

    def update_load(
        self,
        queue_length: int
    ) -> None:
        """Send a load update to the coordinator.

        Args:
            queue_length: Number of unfinished tasks.
        """
        # Update local metrics
        with self._lock:
            if self._queue_length == queue_length:
                return  # No change in load
            self._queue_length = queue_length
        self._send_event(EventType.LOAD_UPDATE)

    def set_status(self, status: StageStatus) -> None:
        """Set the instance status.

        Args:
            status: New status for the instance
        """
        with self._lock:
            if self._status == status:
                return  # No change in status
            self._status = status
        self._send_event(EventType.STATUS_UPDATE)

    def _send_event(self, event_type: EventType) -> None:
        """Send an event to the coordinator.

        Args:
            event_type: Type of the event
            payload: Event-specific payload data
        """
        if not self._dealer_socket:
            raise RuntimeError("Client not started")

        with self._lock:
            message = {
                "event_type": event_type.value,
                "stage_id": self._stage_id,
                "zmq_addr": self.zmq_addr,
                "status": self._status.value,
                "queue_length":self._queue_length,
                "timestamp": time.time(),
            }

        data = OmniSerializer.serialize(message)
        self._dealer_socket.send(data)

    def _heartbeat_loop(self) -> None:
        """Background thread that sends periodic heartbeats."""
        logger.debug(f"Heartbeat thread started for {self.zmq_addr}")
        heartbeat_interval = self._config.heartbeat_interval_ms / 1000.0

        while self._running and not self._stop_event.is_set():
            try:
                self._send_event(EventType.HEARTBEAT)
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")

            self._stop_event.wait(heartbeat_interval)

        logger.debug(f"Heartbeat thread stopped for {self.zmq_addr}")

