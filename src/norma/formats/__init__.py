"""Format capabilities and pure target resolution (WS-02).

The package initialiser is inert: it imports nothing, probes nothing, starts
no process and writes nothing (ARCHITECTURE § 3).

Import public decision functions from ``norma.formats.resolve`` explicitly:
``resolve_target`` produces a provisional C-04 record, ``preflight_targets``
validates all resolvable inputs, and ``finalise_target`` returns a fresh record
after audio planning or a retry. Evidence acquisition stays at the tool edge.
"""
