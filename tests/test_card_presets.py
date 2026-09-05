"""The Card Presets store, as the library dialog now uses it.

The dialog gained an editor, and with it three things the store had to learn: hand out one card's
full values, rename an entry without leaving both copies behind, and describe the fields an editor
should draw. All three are asserted here against a store.json in a temp dir — never the real
library, which is the user's own cards.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import Checker, PACK, fake_package, load_module  # noqa: E402

fake_package("kn", "card_presets", "context")
store = load_module("kn.card_presets.store", "card_presets/store.py")

check = Checker()

TMP = Path(tempfile.mkdtemp(prefix="kinburg_cards_")) / "data" / "store.json"
store._store_path = lambda: str(TMP)          # the real library must not be touched


def seed(presets):
    TMP.parent.mkdir(parents=True, exist_ok=True)
    TMP.write_text(json.dumps({"presets": presets}, ensure_ascii=False), encoding="utf-8")


VASYA = {"type": "character",
         "values": {"name": "Vasya", "gender": "male", "eye_color": "brown", "notes": "quiet"},
         "tags": ["heroes"]}
CAFE = {"type": "entity",
        "values": {"name": "Cafe", "description": "bronze pitchers on every shelf"},
        "tags": ["places"]}

# ------------------------------------------------------------------ the editor's field schema
schema = store.field_schema()
char_keys = [f["key"] for f in schema["character"]]
check("the character schema starts at name and ends at notes",
      char_keys[0] == "name" and char_keys[-1] == "notes", char_keys)
check("...and never offers the two save-only fields as card fields",
      "save_preset_as" not in char_keys and "tags" not in char_keys)
check("...labelled the way the rendered block labels them",
      dict((f["key"], f["label"]) for f in schema["character"]).get("eye_color") == "Eyes")
check("...with the free-form ones marked multiline",
      dict((f["key"], f["multiline"]) for f in schema["character"]).get("notes") is True)
check("...and carrying the node's own tooltip, so the editor explains itself",
      "voice" in dict((f["key"], f["tooltip"]) for f in schema["character"])["voice_tags"].lower())
check("the entity schema is exactly name + description",
      [f["key"] for f in schema["entity"]] == ["name", "description"])

# full_data carries it, so the dialog can write a new card with no extra round-trip
seed({"Vasya": VASYA})
data = store.full_data()
check("full_data ships the schema with the list", set(data["schema"]) == {"character", "entity"})
check("...but not the values — that stays a per-card fetch",
      data["presets"]["Vasya"] == {"type": "character", "tags": ["heroes"]})

# ------------------------------------------------------------------------------ editing values
store.upsert("Vasya", "character", {**VASYA["values"], "eye_color": "green"})
check("an edit rewrites the values", store.get("Vasya")["values"]["eye_color"] == "green")
check("...and leaves the tags alone when none are passed", store.get("Vasya")["tags"] == ["heroes"])
check("...and the block that comes out follows", "- Eyes: green" in store.render("Vasya"))

# ------------------------------------------------------------------------------------- rename
seed({"Vasya": VASYA, "Cafe": CAFE})
store.upsert("Vasiliy", "character", VASYA["values"], old_name="Vasya")
after = store.full_data()["presets"]
check("a rename lands under the new name", "Vasiliy" in after)
check("...and does NOT leave the old one behind", "Vasya" not in after)
check("...inheriting the tags it had", store.get("Vasiliy")["tags"] == ["heroes"])
check("...without disturbing anything else", after["Cafe"] == {"type": "entity", "tags": ["places"]})

# renaming ONTO a name that already exists is an overwrite, not a duplicate
seed({"Vasya": VASYA, "Cafe": CAFE})
store.upsert("Cafe", "character", VASYA["values"], old_name="Vasya")
names = store.full_data()["presets"]
check("renaming onto an existing card replaces it", set(names) == {"Cafe"})
check("...with the moved card's own type", names["Cafe"]["type"] == "character")

# an explicit tag list still wins over the inherited one
seed({"Vasya": VASYA})
store.upsert("Vasiliy", "character", VASYA["values"], tags="band, leads", old_name="Vasya")
check("explicit tags win over the inherited ones", store.get("Vasiliy")["tags"] == ["band", "leads"])

# old_name pointing at nothing is just a create — no crash, no ghost
seed({})
store.upsert("Nobody", "character", {"name": "Nobody"}, old_name="never-existed")
check("a stale old_name degrades to a plain create", list(store.full_data()["presets"]) == ["Nobody"])

# ------------------------------------------------------------- the preview route's own renderer
block = store.render_values("character", {"name": "Zoya", "gender": "female",
                                          "voice_tags": "female lead vocal", "notes": "hums"})
check("unsaved values render the same block a saved card would",
      block.startswith("### Zoya") and "- Voice (music tags): female lead vocal" in block, block)
check("...and a type switch simply ignores the other type's leftovers",
      store.render_values("entity", {"name": "Zoya", "description": "a singer",
                                     "gender": "female"}) == "### Zoya\na singer")
check("an empty card previews as nothing, not as a bare heading",
      store.render_values("character", {"name": "Zoya"}) == "")

check.done()
