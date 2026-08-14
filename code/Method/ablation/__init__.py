def run_sota(*args, **kwargs):
    """Lazily import the configurable parallel ablation runner."""
    from .main_mp import run_sota as run_sota_parallel
    return run_sota_parallel(*args, **kwargs)


def main(argv=None):
    """Run the unified ablation command-line interface."""
    from .interface import main as interface_main
    return interface_main(argv)
