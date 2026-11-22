from .config import MQDepthConfig

__all__ = ["MQDepthConfig", "OutdoorDepth"]


def __getattr__(name):
    if name == "OutdoorDepth":
        from .depth import OutdoorDepth

        return OutdoorDepth
    raise AttributeError(name)
