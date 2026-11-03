"""Errors the library raises. Only an entry point (command line or app page) decides what to do with them."""


class FatalInputError(Exception):
    """Something the user must fix before a run can continue (missing column, no benchmark, library not installed)."""
