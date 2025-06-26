import torch
from typing import Dict, Any

class FisherInformation:
    """Minimal, diagonal Fisher Information estimator.

    This implementation uses the empirical Fisher (squared gradients) and
    Monte-Carlo sampling over mini-batches to approximate the expectation.
    It is intentionally lightweight so it can be called inside a Lightning
    ``training_step`` without heavy overhead.
    """

    # implemented using first order approximation
    # using this paper https://arxiv.org/abs/2107.04205

    def __init__(self, model: torch.nn.Module, monte_carlo_samples: int = 100):
        self.model = model
        self.monte_carlo_samples = max(1, monte_carlo_samples)
        self.reset()

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def reset(self) -> None:
        """Clear stored statistics."""
        self._fisher: Dict[str, torch.Tensor] = {}
        self._sample_count: int = 0

    def accumulate(self) -> None:
        """Accumulate squared gradients currently stored in ``model.parameters``.

        Assumes that ``loss.backward`` (or ``manual_backward`` in Lightning)
        has already populated ``param.grad`` tensors.
        """
        if self.is_done:
            return  # Already gathered desired samples

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            grad_sq = param.grad.detach() ** 2
            if name not in self._fisher:
                # Clone to avoid holding onto the grad tensor reference
                self._fisher[name] = grad_sq.clone()
            else:
                self._fisher[name].add_(grad_sq)

        self._sample_count += 1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_done(self) -> bool:
        """Return True when desired Monte-Carlo samples have been collected."""
        return self._sample_count >= self.monte_carlo_samples

    @property
    def sample_count(self) -> int:
        return self._sample_count

    def get_fisher(self, normalize: bool = True) -> Dict[str, torch.Tensor]:
        """Return the (diagonal) Fisher estimate.

        Args:
            normalize: If True (default) divide by the number of samples so the
                returned values approximate the expectation.
        """
        if self._sample_count == 0:
            return {}
        if normalize:
            return {k: v / self._sample_count for k, v in self._fisher.items()}
        return {k: v.clone() for k, v in self._fisher.items()}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """Return a python dict for serialization."""
        return {
            "fisher": {k: v.cpu() for k, v in self._fisher.items()},
            "sample_count": self._sample_count,
            "monte_carlo_samples": self.monte_carlo_samples,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.monte_carlo_samples = int(state_dict.get("monte_carlo_samples", self.monte_carlo_samples))
        self._sample_count = int(state_dict.get("sample_count", 0))
        fisher_cpu = state_dict.get("fisher", {})
        self._fisher = {k: v.clone() for k, v in fisher_cpu.items()} 