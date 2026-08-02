# synth_native package – re-exports the compiled extension if available.
try:
    from synth_native._fm_synth import FMSynth  # noqa: F401
except ImportError:
    pass  # not yet built – model/fm_synth.py will fall back to pure Python
