from typing import Tuple

from numpy.typing import NDArray


class BaseAgent(object):
    def __init__(self, **kwargs):
        super(BaseAgent, self).__init__(**kwargs)

    def train(
        self,
        ob_batch: NDArray,
        ac_batch: NDArray,
        re_batch: NDArray,
        next_ob_batch: NDArray,
        terminal_batch: NDArray,
    ) -> dict:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def add_to_replay_buffer(self, paths: "list[dict]"):
        raise NotImplementedError

    def sample(
        self, batch_size: int
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        raise NotImplementedError

    def save(self, path: dict):
        raise NotImplementedError
