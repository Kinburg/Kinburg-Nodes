"""Local LLM Chat (GGUF) — a self-contained multi-turn chat node.

The node itself is deliberately bare: just the **chat window**, the persona chips and the
**Send / Approve / Clear** buttons. Everything about *how* to generate — model, system prompt,
sampling, loader options, reasoning split, output format, and the optional vision inputs — comes in
through a **`Local LLM Settings (GGUF)`** bundle on ``persona_1``.

Interaction (buttons live in web/chat_llm.js):
  • **📨 Send** runs the workflow up to this node: ``run()`` generates a reply from the stored
    history + your message (+ image), shows it, and BLOCKS the downstream branch
    (``ExecutionBlocker``) so nothing past the node runs while you chat.
  • **✅ Approve** runs with the gate open: ``run()`` skips generation and emits the **last reply**
    on ``text``, so it flows downstream immediately (no re-generation, any seed). With
    ``unload_on_approve`` (default on) it also frees the LLM from VRAM first, so the image model
    that comes next has room.

**`chat_state`** is the node's whole client-side state as one JSON string — the conversation, your
pending message, the picked persona, the Approve flag and the turn descriptor. It is ONE input on
purpose: the Vue frontend draws a 24 px row for every widget a node has, whether or not the widget
is hidden, so six little carriers cost 168 px of dead grey space under the chat. The frontend
doesn't even render this one as a widget — it splices the auto-created widget out and lets the chat
window itself carry the value (an `addDOMWidget` with `getValue`/`setValue`), so the node has no
hidden rows at all. Shape::

    {"v": 1, "user": str, "history": [...], "nonce": int, "approved": bool,
     "persona": 1..6, "att": [...],
     "turn": {"mode": ..., "persona": ..., "keep": {...}, "private": [...]}}

**Attachments.** ``att`` is the tray: pictures staged for the NEXT send, each a ComfyUI file ref
``{name, subfolder, type}`` (plus optional ``caption`` and ``shot``) — the very triple
``/upload/image`` returns and ``/view`` takes, so the chat window shows the thumbnail and this node
opens the same file. Deliberately a reference and not a graph link: 📨 Send re-runs everything
UPSTREAM of this node, so an IMAGE wired in would drag its whole generation branch along on every
message. On send the frontend moves the tray onto the message it belongs to (``m["att"]``), where
it stays for good — the chat keeps showing the picture.

The model's copy does not. Pixels ride the turn they were attached on and nothing else; from the
next turn the image is a text marker (see ``_content_of``). One picture costs 700-2500 tokens and
llama.cpp clears the KV cache on every vision call, so a handful of live images would both eat the
window and be re-encoded every turn. Alternating between turns with and without a picture is
cheap: the projector is not part of the load signature, so the worker attaches it for the turn
that needs it and releases its clip on the next one, while the model itself stays loaded.

**Personas.** Up to six ``persona_1..6`` inputs each take a WHOLE settings bundle, so a persona
brings its own model, sampling and system prompt. ``persona_1`` is required and doubles as the
node's plain config — wire only that one and this is an ordinary chat node with no chip row. The
active persona supplies the whole config for that turn. Personas that share a model and loader
settings cost no reload, since the worker's load signature ignores the system prompt and sampling.

**Turn modes** (``turn.mode``):
  • ``turn`` — you typed something; a normal user message.
  • ``fresh`` — the box was empty and you switched persona: NO user message is sent at all, so a
    persona whose whole job is in its system prompt isn't prodded with a throwaway instruction
    that everyone else would then see in the context.
  • ``continue`` — the box was empty and the last reply is the active persona's own: resume that
    reply where max_tokens cut it off (a prefill in the worker), appending to the same message.
  • ``fold`` — **archiving**. Not a chat turn: the older part of the conversation (everything up to
    ``turn["upto"]``) is handed to a summariser as a one-shot task and comes back as a *brief*,
    which the frontend stores as a single ``{"role": "digest"}`` entry while the messages it
    replaced get a ``"fold"`` marker. From then on those messages stay in the chat, dimmed, but the
    model sees the brief instead — appended to the system prompt through the same ``context`` slot a
    Character Card uses. The brief is cumulative: each pass rewrites it from the previous one plus
    the next block, so archiving twice never loses the first pass. Nothing is deleted, and deleting
    the digest bubble puts the originals back. The summariser is a persona picked in ⚙, or — by
    default — the active persona's model with ``DIGEST_SYSTEM`` swapped in, which costs no reload
    since that model is already resident. Turns already withheld (by hand, by a retention window,
    or by a private persona) are never folded in: a private thread must not leak into a shared
    brief.
``fresh``/``continue`` suppress ``thinking_directive`` — it would become the user turn we are
deliberately not sending.

**What reaches the model.** Three things withhold a message — it stays visible (dimmed) in the chat
either way, and all of it is decided here, per request, rather than stamped into the history, so it
follows the settings instead of freezing whatever they were when the turn happened:
  • ``"ctx": false`` on the message — hidden by hand, cleared with the 👁 button.
  • A **retention window** (*how long*): ``turn["keep"]`` maps a persona's name to how many of its
    most recent turns survive. A prompt-writer that emits a 300-token draft every press is only
    useful for an iteration or two, so ``{"Generator": 2}`` ages its older drafts out; a persona
    with no entry keeps everything, ``0`` keeps none.
  • **Privacy** (*for whom*): ``turn["private"]`` lists personas whose turns only they themselves
    get to see. The prompt-writer still reads its own last drafts to revise them, while the persona
    you're brainstorming with never has to wade through them.
The two are orthogonal — ``private`` + ``keep: 2`` is "I see my own last two, nobody else sees any".

Replies made by a *different* persona than the one now speaking are prefixed with that persona's
name, so the model can tell someone else's turns from its own.
"""
import json
import os

from .attachments import resolve_refs
from .llm_node import HELP_TEXT, build_llm_request, _generate_and_format, _shutdown_worker
from .settings_node import LLM_CONFIG
from ..categories import CAT_LLM

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except Exception:  # pragma: no cover - only inside ComfyUI
    ExecutionBlocker = None

MODES = ("turn", "fresh", "continue", "fold")

DIGEST_SYSTEM = """You compress a working conversation into a compact brief that another assistant \
will act on.

Preserve every concrete decision and constraint EXACTLY as stated — names, numbers, counts, \
colours, aspect ratios, and anything the user asked to avoid. Drop pleasantries, restatements and \
superseded drafts. Never invent, never soften a requirement, never add suggestions of your own.

Reply with the brief and nothing else, in this shape, omitting any heading that has nothing under \
it:

## Subject
## Constraints
## Style
## Decided
## Open questions"""

# Header the digest wears inside the system prompt, so the model knows it is looking at a summary
# of turns it can no longer see rather than at instructions.
DIGEST_HEADER = "## Earlier in this conversation (summary of archived turns)"


def _speaker_of(m):
    if m.get("role") == "user":
        return "User"
    return str(m.get("persona") or "Assistant")


def _content_of(m):
    """A message as the MODEL sees it: its text plus a text stand-in for anything attached.

    Pixels only ever ride the turn they were attached on (see ``_load_attachments``), so from the
    next turn onwards an image is remembered by this line alone — one image is worth 700-2500
    tokens and llama.cpp clears the KV cache on every vision call, so keeping them live would cost
    the whole window and re-encode the lot each turn. The picture itself stays in the chat forever;
    only the model's copy degrades to text. Built per request, never stamped into the history, so
    editing a caption changes what the model reads about pictures already sent.
    """
    text = m.get("content") if isinstance(m.get("content"), str) else ""
    marks = []
    for a in (m.get("att") or []):
        if not isinstance(a, dict):
            continue
        # ``ctx: false`` — visible in the chat, invisible to the model, the same convention a
        # message-level ``ctx: false`` follows. Send Image to Chat stamps it on a persona's
        # picture: the persona usually described the scene a line earlier, so a marker would say
        # it twice, and it never gets to look at the picture anyway.
        if a.get("ctx") is False:
            continue
        cap = str(a.get("caption") or "").strip()
        marks.append("[image: " + cap + "]" if cap else "[image]")
    if not marks:
        return text
    joined = " ".join(marks)
    return (text + "\n" + joined) if text else joined


def _fold_prompt(previous, block):
    """The summariser's single-shot task: fold `block` into the brief we already have."""
    convo = "\n\n".join(f"{_speaker_of(m)}: {_content_of(m)}" for m in block)
    prev = (previous or "").strip() or "(nothing yet — this is the first pass)"
    return ("Brief so far — update it, don't start over:\n\n" + prev
            + "\n\n---\n\nNext part of the conversation to fold into it:\n\n" + convo
            + "\n\n---\n\nReturn the updated brief.")


def _digest_of(history):
    for m in history:
        if isinstance(m, dict) and m.get("role") == "digest":
            return str(m.get("content") or "")
    return ""


def _stream_cb(unique_id):
    """Push each generated token to the chat window over ComfyUI's websocket, or None outside it."""
    try:
        from server import PromptServer
        nid = str(unique_id[0] if isinstance(unique_id, list) else unique_id)

        def token_cb(delta):
            try:
                PromptServer.instance.send_sync("kinburg.chatllm", {"id": nid, "delta": delta})
            except Exception:
                pass
        return token_cb
    except Exception:
        return None


def _load_attachments(att):
    """(images, error) — one [1,H,W,C] float tensor per attachment, ComfyUI's IMAGE layout.

    Separate tensors rather than one batch: the files are whatever you pasted or generated and
    need not share a resolution. `_encode_images` takes the list as-is.
    """
    if not att:
        return [], ""
    paths, missing = resolve_refs(att)
    if missing:
        return None, ("attachment file is gone: " + ", ".join(sorted(set(missing)))
                      + " — re-attach it (ComfyUI clears its temp folder on restart).")
    import numpy as np
    import torch
    from PIL import Image
    out = []
    for p in paths:
        try:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        except Exception as e:
            return None, f"cannot read attachment '{os.path.basename(p)}': {e}"
        out.append(torch.from_numpy(arr)[None, ...])
    return out, ""


def _last_assistant(history):
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def _windowed_out(history, keep):
    """Indices withheld by the per-persona retention windows.

    A persona's *turn* is its assistant message plus the user message that prompted it (a turn sent
    with no user message is just the reply). Only the persona's ``keep`` most recent turns stay in
    the context; older ones age out for everybody. ``keep`` of 0 withholds all of them, a missing
    or non-integer entry means "keep everything". Counting a persona's OWN turns — not every turn
    since — is deliberate: chatting with someone else for twenty messages shouldn't make the
    prompt-writer forget the draft you're iterating on.

    Mirrored in web/chat_llm.js (windowedOut) so the chat dims exactly what this drops.
    """
    out = set()
    if not isinstance(keep, dict) or not keep:
        return out
    turns = {}
    for i, m in enumerate(history):
        if isinstance(m, dict) and m.get("role") == "assistant":
            who = str(m.get("persona") or "")
            if who:
                turns.setdefault(who, []).append(i)
    for who, idxs in turns.items():
        n = keep.get(who)
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            continue
        for i in (idxs[:-n] if n else idxs):
            out.add(i)
            prev = history[i - 1] if i > 0 else None
            if (isinstance(prev, dict) and prev.get("role") == "user"
                    and str(prev.get("persona") or "") == who):
                out.add(i - 1)          # the instruction that produced it goes with it
    return out


def _private_out(history, private, speaker):
    """Indices withheld because their persona keeps its turns to itself.

    A private persona still reads its OWN turns — that's what makes iterating on a draft work; it
    is everyone else who never sees them. Both halves of the turn go (the reply and the instruction
    that produced it), since both carry the persona tag. Untagged messages are always shared.

    Mirrored in web/chat_llm.js (privateOut). Unlike the retention window this depends on who is
    speaking, so the chat's dimming changes as you switch chips — which is the point: it shows what
    the persona you picked is about to see.
    """
    out = set()
    names = {str(x) for x in private} if isinstance(private, (list, tuple, set)) else set()
    if not names:
        return out
    for i, m in enumerate(history):
        if not isinstance(m, dict):
            continue
        who = str(m.get("persona") or "")
        if who and who != speaker and who in names:
            out.add(i)
    return out


def _parse_state(raw):
    """chat_state -> (history, user_message, approved, active_persona, turn). Never raises: a
    truncated or hand-edited state degrades to an empty chat instead of breaking the graph."""
    try:
        st = json.loads(raw) if raw else {}
    except Exception:
        st = {}
    if not isinstance(st, dict):
        st = {}
    history = st.get("history")
    turn = st.get("turn")
    att = st.get("att")
    return (
        history if isinstance(history, list) else [],
        st.get("user") if isinstance(st.get("user"), str) else "",
        bool(st.get("approved")),
        st.get("persona"),
        turn if isinstance(turn, dict) else {},
        att if isinstance(att, list) else [],
    )


_PERSONA_TIP = ("Persona #{n}: wire a whole 'Local LLM Settings (GGUF)' here — its system prompt, "
                "model and sampling become the config for every turn this persona speaks. A chip "
                "for it appears in the chat as soon as a second persona is wired. Keep the loader "
                "fields (model, n_ctx, n_gpu_layers, flash_attn, kv_cache_type) identical across "
                "personas and switching between them costs no model reload.")


class LocalLLMChatGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # persona_1 IS the node's config — with nothing else wired the chat behaves like
                # any other LLM node, and the chip row stays hidden.
                "persona_1": (LLM_CONFIG, {"tooltip": _PERSONA_TIP.format(n=1) + " This one is also the node's default config."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional image for vision, attached to every turn while it stays connected. Prefer pasting or dropping a picture straight into the chat window instead — that attaches it to ONE turn and doesn't re-run this branch on every message. Needs an mmproj on the active Settings node either way."}),
                "persona_2": (LLM_CONFIG, {"tooltip": _PERSONA_TIP.format(n=2)}),
                "persona_3": (LLM_CONFIG, {"tooltip": _PERSONA_TIP.format(n=3)}),
                "persona_4": (LLM_CONFIG, {"tooltip": _PERSONA_TIP.format(n=4)}),
                "persona_5": (LLM_CONFIG, {"tooltip": _PERSONA_TIP.format(n=5)}),
                "persona_6": (LLM_CONFIG, {"tooltip": _PERSONA_TIP.format(n=6)}),
                "unload_on_approve": ("BOOLEAN", {"default": True, "tooltip": "Free the LLM from VRAM when you press ✅ Approve, so the image model downstream has room. Turn off to keep chatting at full speed when nothing heavy follows."}),
                # The whole chat, carried by the chat window itself — see the module docstring.
                # The frontend removes the widget ComfyUI auto-creates for this, so it costs no row.
                "chat_state": ("STRING", {"default": "", "tooltip": "The conversation, managed by the chat window. You don't edit this directly."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "help")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT_LLM

    def run(self, persona_1, image=None, persona_2=None, persona_3=None, persona_4=None,
            persona_5=None, persona_6=None, unload_on_approve=True, chat_state="",
            unique_id=None):
        history, user_message, approved, active_persona, turn, att = _parse_state(chat_state)

        # ── Approve: release the last saved reply downstream, no generation. ──────────────
        # Deliberately ignores `ctx: false` — a muted persona's reply is exactly what you approve.
        if approved:
            if unload_on_approve:
                _shutdown_worker()
            return (_last_assistant(history), HELP_TEXT)

        def _blocked(reply, thoughts="", stats=None):
            gate = ExecutionBlocker(None) if ExecutionBlocker is not None else reply
            payload = json.dumps({"reply": reply, "thoughts": thoughts or "", "stats": stats or {}},
                                 ensure_ascii=False)
            return {"ui": {"kinburg_chatllm": [payload]}, "result": (gate, HELP_TEXT)}

        mode = turn.get("mode") if turn.get("mode") in MODES else "turn"
        speaker = str(turn.get("persona") or "")
        keep = turn.get("keep") if isinstance(turn.get("keep"), dict) else {}
        private = turn.get("private") if isinstance(turn.get("private"), list) else []

        # The active persona's whole bundle IS the config for this turn. An out-of-range or
        # unwired pick falls back to the first persona that is actually connected.
        personas = (persona_1, persona_2, persona_3, persona_4, persona_5, persona_6)
        try:
            pi = int(active_persona or 0)
        except (TypeError, ValueError):
            pi = 0
        cfg = (personas[pi - 1] if (1 <= pi <= len(personas)
                                    and isinstance(personas[pi - 1], dict)) else None)
        if cfg is None:
            cfg = next((p for p in personas if isinstance(p, dict)), None)
        if cfg is None:
            return _blocked("[ERROR] No settings connected — wire a 'Local LLM Settings (GGUF)' "
                            "node into persona_1.")

        # Windows/privacy also decide what may be archived — a private persona's turns must not
        # leak into a shared brief, and anything already withheld is not worth summarising.
        # speaker="" here means "nobody's own turns", so every private persona is excluded.
        shared_out = _windowed_out(history, keep) | _private_out(history, private, "")

        # ── Archive: fold the older turns into one brief, no chat turn at all. ────────────
        if mode == "fold":
            try:
                upto = max(0, min(int(turn.get("upto") or 0), len(history)))
            except (TypeError, ValueError):
                upto = 0
            block = [m for i, m in enumerate(history[:upto])
                     if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                     and _content_of(m)
                     and m.get("ctx") is not False and not m.get("fold") and i not in shared_out]
            if not block:
                return _blocked("[ERROR] Nothing left to archive in that range.")

            try:
                si = int(turn.get("summarizer") or 0)
            except (TypeError, ValueError):
                si = 0
            scfg = (personas[si - 1] if (1 <= si <= len(personas)
                                         and isinstance(personas[si - 1], dict)) else None)
            # A persona picked as the summariser speaks in its own voice; otherwise borrow the
            # active persona's model (already loaded, so no reload) and override the prompt.
            override = None if scfg is not None else DIGEST_SYSTEM
            err, ctx = build_llm_request(scfg if scfg is not None else cfg,
                                         _fold_prompt(_digest_of(history), block),
                                         system_override=override)
            if err:
                return _blocked(f"[ERROR] {err}")
            ctx["req"]["stream_text"] = True
            fstats = {}
            fout = _generate_and_format(ctx["req"], ctx["load_sig"], ctx["max_tokens"],
                                        ctx["unload_comfy"], ctx["unload_llm"], ctx["directive"],
                                        True, ctx["answer_marker"], ctx["help"],
                                        token_cb=_stream_cb(unique_id), stats=fstats)
            if fstats:
                fstats["seconds"] = fout[6]
            payload = json.dumps({"reply": fout[0], "thoughts": fout[1], "stats": fstats,
                                  "fold": {"upto": upto, "count": len(block)}},
                                 ensure_ascii=False)
            gate = ExecutionBlocker(None) if ExecutionBlocker is not None else fout[0]
            return {"ui": {"kinburg_chatllm": [payload]}, "result": (gate, HELP_TEXT)}

        # 'continue' pulls the truncated reply out of the history and hands it to the worker as a
        # prefill instead of as a message. Fall back to 'fresh' if it isn't where we expect.
        src, cont = history, ""
        if mode == "continue":
            last = history[-1] if history else None
            if (isinstance(last, dict) and last.get("role") == "assistant"
                    and isinstance(last.get("content"), str) and last["content"]):
                cont, src = last["content"], history[:-1]
            else:
                mode = "fresh"

        # Windows are measured over the WHOLE history, then applied to `src` (a prefix of it), so
        # lifting the last reply out for a continuation can't shift anybody's turn count.
        dropped = _windowed_out(history, keep) | _private_out(history, private, speaker)

        clean_history = []
        for i, m in enumerate(src):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            # hidden by hand / archived into the brief / withheld by a window or privacy
            if m.get("ctx") is False or m.get("fold") or i in dropped:
                continue
            # An older turn's picture is down to its text marker by now, which is also what keeps
            # a message that was NOTHING but an image from vanishing out of the context.
            content = _content_of(m)
            if not content:
                continue
            who = str(m.get("persona") or "")
            if role == "assistant" and who and who != speaker:
                content = f"[{who}]: {content}"  # someone else's turn, not the speaker's own
            clean_history.append({"role": role, "content": content})

        # The brief stands in for the turns it archived: it goes in as reference material on the
        # system prompt (the same slot a Character Card uses), not as a message.
        digest = _digest_of(history)
        if digest.strip():
            cfg = dict(cfg)
            cfg["context"] = "\n\n".join(x for x in [
                (cfg.get("context") or "").strip(), DIGEST_HEADER + "\n" + digest.strip()] if x)

        # Without a user turn there is nowhere to put a reasoning directive except a message we
        # are deliberately not sending, so drop it for these two modes.
        if mode != "turn" and cfg.get("thinking_directive", "model default") != "model default":
            cfg = dict(cfg)
            cfg["thinking_directive"] = "model default"

        # This turn's pixels: whatever sits in the tray, plus the legacy `image` socket if it is
        # wired. A 'continue' is a prefill on the raw completion API, which the vision path cannot
        # do (the worker refuses it), so that mode sends none. Empty -> None, NOT an empty list:
        # `build_llm_request` treats "not None" as "turn vision on", and asking for the projector
        # with nothing to look at would load it for nothing.
        turn_images = []
        if mode != "continue":
            loaded, aerr = _load_attachments(att)
            if aerr:
                return _blocked(f"[ERROR] {aerr}")
            if image is not None:
                turn_images.append(image)
            turn_images.extend(loaded)

        err, ctx = build_llm_request(cfg,
                                     user_message if mode == "turn" else "",
                                     image=turn_images or None,
                                     history=clean_history)
        if err:
            return _blocked(f"[ERROR] {err}")
        # Opt in to the worker's "no final user message" path — every other node keeps the old
        # behaviour of always sending one, even when the prompt is empty.
        ctx["req"]["allow_no_user"] = mode != "turn"
        if cont:
            ctx["req"]["continue_text"] = cont

        # Stream the reply text to the node as it generates, over ComfyUI's websocket.
        ctx["req"]["stream_text"] = True
        token_cb = _stream_cb(unique_id)

        # Always split reasoning out (strip_think=True) so the chat shows a clean answer + a
        # separate 'thoughts' block, regardless of the config's strip_think. `stats` comes back
        # filled with the token / context-fill figures the chat's meter shows.
        stats = {}
        out = _generate_and_format(ctx["req"], ctx["load_sig"], ctx["max_tokens"],
                                   ctx["unload_comfy"], ctx["unload_llm"], ctx["directive"],
                                   True, ctx["answer_marker"], ctx["help"], token_cb=token_cb,
                                   stats=stats)
        if stats:
            stats["seconds"] = out[6]
        return _blocked(out[0], out[1], stats)  # (answer, thoughts, stats)


NODE_CLASS_MAPPINGS = {"LocalLLMChatGGUF": LocalLLMChatGGUF}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMChatGGUF": "Local LLM Chat (GGUF)"}
