"""The wildcard type used across the pack for ComfyUI's "accept any link" idiom.

ComfyUI type-checks links by string equality, so a type string that compares equal to
everything makes an input/output accept any connection. Import ``ANY`` (or ``AnyType``) from
here instead of redefining it per module.
"""


class AnyType(str):
    """A type string that compares equal to everything, so ComfyUI's link type-check passes."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

    # str overrides __eq__ -> it's unhashable unless we restore a hash; keep the str hash so the
    # instance still works as a dict key / in sets (ComfyUI stores types around).
    __hash__ = str.__hash__


# The single shared instance. `_AnyType` alias eases dropping into modules that used that name.
ANY = AnyType("*")
_AnyType = AnyType
