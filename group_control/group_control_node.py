"""Group Control — a client-side control panel for toggling LiteGraph groups by name.

This node does no work at execution time: it declares no inputs and no outputs, so it is never
part of the prompt and never runs on the backend. Everything lives in web/group_control.js,
which reads the workflow's groups, lists each *unique* group name, and lets you flip every
group carrying that name between **Always** (active) and **Bypass** (skipped) by rewriting the
`mode` of the nodes inside those groups before the prompt is built.

Design notes for the frontend side:
  * Same-named groups are controlled together (dedup by title).
  * Group membership is resolved by bounding box, so an outer group automatically covers any
    groups nested inside it; nested names are shown indented by depth.
  * The panel polls the graph and grows/rebuilds itself as groups are added, renamed or removed.
  * No per-node state is stored here — the on/off state is the `mode` already saved on the
    target nodes, so it survives a workflow reload for free.
"""


class KinburgGroupControl:
    """UI-only node. All behaviour is in web/group_control.js."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    CATEGORY = "Kinburg-Nodes/util"
    OUTPUT_NODE = False
    DESCRIPTION = (
        "Control panel for enabling/bypassing workflow groups by name. Groups sharing a name "
        "are toggled together; nested groups are supported. The list grows automatically as "
        "groups are added to the workflow."
    )

    def noop(self):
        # Never actually called (the node has no outputs and is excluded from the prompt),
        # but ComfyUI requires FUNCTION to point at a real method.
        return ()


NODE_CLASS_MAPPINGS = {"KinburgGroupControl": KinburgGroupControl}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgGroupControl": "Group Control 🎚️"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
