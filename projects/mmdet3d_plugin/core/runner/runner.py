from collections import OrderedDict

from mmengine.runner import Runner
from mmengine.registry import RUNNERS
from mmengine.utils.dl_utils import collect_env


__all__ = ['CustomRunner']

@RUNNERS.register_module()
class CustomRunner(Runner):

    def _log_env(self, env_cfg: dict) -> None:
        """Logging environment information of the current task.

        Args:
            env_cfg (dict): The environment config of the runner.
        """
        # Collect and log environment information.
        env = collect_env()
        runtime_env = OrderedDict()
        runtime_env.update(env_cfg)
        runtime_env.update(self._randomness_cfg)
        runtime_env['Distributed launcher'] = self._launcher
        runtime_env['Distributed training'] = self._distributed
        runtime_env['GPU number'] = self._world_size

        env_info = '\n    ' + '\n    '.join(f'{k}: {v}'
                                            for k, v in env.items())
        runtime_env_info = '\n    ' + '\n    '.join(
            f'{k}: {v}' for k, v in runtime_env.items())
        dash_line = '-' * 60
        self.logger.info('\n' + dash_line + '\nSystem environment:' +
                         env_info + '\n'
                         '\nRuntime environment:' + runtime_env_info + '\n' +
                         dash_line + '\n')