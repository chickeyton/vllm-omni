# Experimental Full-Duplex (JoyVL)

This package now contains only the JoyVL framework and its example
integration:

```text
core/   generic duplex scaffold used by the JoyVL adapter
joyvl/  JoyVL model-specific integration
```

To run JoyVL, see
[`recipes/JD/JoyAI-VL-Interaction.md`](../../../recipes/JD/JoyAI-VL-Interaction.md).

The MiniCPM-o 4.5 native full-duplex runtime graduated out of this package.
It now lives in the stable tree:

```text
vllm_omni/engine/duplex/                       engine control plane, sessions, leases
vllm_omni/entrypoints/openai/duplex/           WebSocket serving and Realtime projection
vllm_omni/entrypoints/duplex_request_client.py request/output lifecycle
vllm_omni/model_executor/models/minicpmo_4_5/duplex/  MiniCPM adapter
vllm_omni/model_executor/duplex_sampling.py    AR-runner sampling hook helper
vllm_omni/outputs/duplex.py                    typed output decision envelope
```

For its architecture and validation scope, see
[`docs/design/fullduplex.md`](../../../docs/design/fullduplex.md).
