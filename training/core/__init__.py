"""核心训练组件包。

保持这里轻量，避免 `training.core.config` 这类轻量导入被 `torch` 等训练依赖拖入。
需要的符号请从具体子模块直接导入，例如：
`from training.core.config import ExperimentConfig`
"""

__all__: list[str] = []
