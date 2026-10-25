"""Bound CPU thread pools so vision and mapping do not oversubscribe a laptop."""

# Allow this entry point to run directly or as a package module.
if __package__ in (None, ''):
    import sys as _entry_sys
    from pathlib import Path as _EntryPath
    _entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import os

_configured = None
_thread_limiter = None


def configure_compute_runtime():
    global _configured, _thread_limiter
    if _configured is not None:
        return dict(_configured)
    threads = max(1, min(8, int(os.environ.get('LVIMOT_CPU_THREADS', '2'))))
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(name,str(threads))
    result = {'cpu_threads':threads,'opencv_threads':1,'cuda_available':False}
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
        result.update({'torch':torch.__version__,'cuda_available':bool(torch.cuda.is_available())})
        # Avoid large autotuning workspaces while CARLA shares a 4 GB GPU.
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_limits
        _thread_limiter = threadpool_limits(limits=threads)
        result['native_thread_pools_bounded'] = True
    except ImportError:
        result['native_thread_pools_bounded'] = False
    _configured = result
    return dict(result)


def configure_detector_threads():
    """Ultralytics otherwise resets CPU inference to its own thread count."""
    threads=max(1,min(8,int(os.environ.get('LVIMOT_CPU_THREADS','2'))))
    from ultralytics.utils import torch_utils
    torch_utils.NUM_THREADS=threads
    import torch
    torch.set_num_threads(threads)
