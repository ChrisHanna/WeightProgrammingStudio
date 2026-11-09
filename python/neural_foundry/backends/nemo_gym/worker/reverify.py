"""Run NeMo reverification with explicit resource-server shutdown."""

import atexit

from nemo_gym.cli.env import RunHelper

original_start = RunHelper.start


def managed_start(self, *args, **kwargs):
    original_start(self, *args, **kwargs)
    atexit.register(self.shutdown)


RunHelper.start = managed_start

from nemo_gym.cli.main import main

main()