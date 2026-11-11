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

# ----------------------------------------------------------------------------------- pictures
# A card can keep one picture (a character sheet, a location) for reference-image video models.
import base64  # noqa: E402
import io  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

PICS = TMP.parent / "pictures"


def files():
    return sorted(p.name for p in PICS.glob("*.png")) if PICS.is_dir() else []


sheet = Image.new("RGB", (40, 20), (200, 30, 10))
seed({"Vasya": VASYA, "Cafe": CAFE})
store.upsert("Vasya", "character", VASYA["values"], picture=sheet)
pic = store.get("Vasya").get("picture", "")
check("a picture is stored next to the library, under a hash name",
      files() == [pic] and len(pic) == 20 and pic.endswith(".png"), files())
check("...and the card points at it", store.picture_path(pic) == str(PICS / pic))
check("the list tells the dialog which cards have one",
      store.full_data()["presets"]["Vasya"].get("picture") == pic
      and "picture" not in store.full_data()["presets"]["Cafe"])
check("a card without a picture is listed exactly as before",
      store.full_data()["presets"]["Cafe"] == {"type": "entity", "tags": ["places"]})

store.upsert("Vasya", "character", {**VASYA["values"], "eye_color": "grey"})
check("an edit that sends no picture keeps it", store.get("Vasya").get("picture") == pic)
store.retag("Vasya", "heroes, band")
check("...and so does a retag", store.get("Vasya").get("picture") == pic)
store.upsert("Vasiliy", "character", VASYA["values"], old_name="Vasya")
check("a rename carries the picture along", store.get("Vasiliy").get("picture") == pic)

store.upsert("Vasiliy copy", "character", VASYA["values"], picture=pic)
check("a duplicate shares the same file instead of copying it",
      store.get("Vasiliy copy").get("picture") == pic and files() == [pic])
store.upsert("Vasiliy copy 2", "character", VASYA["values"], picture=Image.new("RGB", (40, 20), (200, 30, 10)))
check("the same pixels saved again land on the same file", files() == [pic], files())

store.upsert("Vasiliy", "character", {}, delete=True)
check("deleting one card keeps a picture another still uses", files() == [pic])
store.upsert("Vasiliy copy", "character", {}, delete=True)
store.upsert("Vasiliy copy 2", "character", VASYA["values"], picture="")
check("removing the last reference deletes the file",
      files() == [] and "picture" not in store.get("Vasiliy copy 2"), files())

store.upsert("Cafe", "entity", CAFE["values"], picture=Image.new("RGB", (8, 8), (0, 0, 255)))
first = store.get("Cafe")["picture"]
store.upsert("Cafe", "entity", CAFE["values"], picture=Image.new("RGB", (8, 8), (0, 255, 0)))
check("replacing a picture deletes the old file",
      files() == [store.get("Cafe")["picture"]] and first not in files(), files())

refused = [bad for bad in ("../store.json", "..\\store.json", "abc.png", first.upper(), "", None, 5)
           if store.picture_path(bad) is None]
check("picture_path answers only for its own hash names — never a path the browser made up",
      len(refused) == 7, refused)
check("...and not for a picture that has been deleted", store.picture_path(first) is None)
try:
    store.upsert("Cafe", "entity", CAFE["values"], picture="0123456789abcdef.png")
    check("a picture name that isn't stored is refused", False)
except ValueError:
    check("a picture name that isn't stored is refused", True)

buf = io.BytesIO()
Image.new("RGBA", (6, 4), (10, 20, 30, 128)).save(buf, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
img = store.picture_from_data_url(url)
check("the dialog's data: URL decodes to the picture", img.size == (6, 4))
for bad in ("data:text/plain;base64,aGk=", "data:image/png;base64,bm90IGFuIGltYWdl", "nonsense"):
    try:
        store.picture_from_data_url(bad)
        check(f"a broken picture is refused: {bad[:24]}", False)
    except ValueError:
        pass
check("a broken or non-image data: URL is a ValueError (the route answers 400)", True)

t = torch.zeros(2, 4, 6, 3)
t[0, :, :, 0] = 1.0
t[1] = 0.5
p = store.picture_from_tensor(t)
check("an IMAGE batch keeps only its first frame",
      p.size == (6, 4) and p.getpixel((0, 0)) == (255, 0, 0), p.getpixel((0, 0)))

# ---------------------------------------------------------------- the nodes that save and read
cc = load_module("kn.context.character_card", "context/character_card.py")
seed({})
cc.CharacterCard().run(name="Keen", gender="female", save_preset_as="Keen", picture=t)
kp = store.get("Keen").get("picture")
check("Character Card saves the picture wired into it", bool(store.picture_path(kp)))
cc.CharacterCard().run(name="Keen", gender="female", hair_color="red", save_preset_as="Keen")
check("...and a re-save without one keeps it", store.get("Keen").get("picture") == kp)
check("the editor's schema doesn't grow a 'picture' text field",
      "picture" not in [f["key"] for f in store.field_schema()["character"]])

ec = load_module("kn.context.entity_card", "context/entity_card.py")
ec.EntityCard().run(name="Kharkiv", description="ruins", save_preset_as="Kharkiv", picture=t[1:])
check("Entity Card saves one too", bool(store.picture_path(store.get("Kharkiv").get("picture"))))

sn = load_module("kn.card_presets.save_node", "card_presets/save_node.py")
_, saved_as, report = sn.CardSave().run('{"name": "Burg", "gender": "male"}', picture=t)
check("Card Save stores the picture and says so",
      saved_as == "Burg" and "with picture" in report and bool(store.get("Burg").get("picture")), report)

pn = load_module("kn.card_presets.presets_node", "card_presets/presets_node.py")
card, voice, out, cast = pn.CardPresets().run("Keen")
check("Card Presets gives the picture back as an IMAGE batch of one",
      tuple(out.shape) == (1, 4, 6, 3) and out.dtype == torch.float32, tuple(out.shape))
check("...with the pixels it was saved with", torch.allclose(out[0], t[0]))
check("...None for a card without one, and for no card at all",
      pn.CardPresets().run("Burg")[2] is not None and pn.CardPresets().run(store.NONE)[2] is None)
store.upsert("Burg", "character", store.get("Burg")["values"], picture="")
check("...and None once the picture is removed", pn.CardPresets().run("Burg")[2] is None)
check("the cast output is the whole card, for Morpheus",
      cast["name"] == "Keen" and cast["type"] == "character" and cast["values"]["gender"] == "female"
      and torch.equal(cast["picture"], out))
check("...and nothing at all for no card", pn.CardPresets().run(store.NONE)[3] is None)
check("the node gained no widget — its saved buttons would land in it",
      list(pn.CardPresets.INPUT_TYPES()["required"]) == ["preset"]
      and list(pn.CardPresets.INPUT_TYPES()["optional"]) == ["filter"])
check("an entity card goes out as an entity", pn.CardPresets().run("Kharkiv")[3]["type"] == "entity")
store.upsert("filed-as", "character", {"name": "", "gender": "male"})
check("a card with an empty name field is named by its library key",
      pn.CardPresets().run("filed-as")[3]["name"] == "filed-as")
before = pn.CardPresets.IS_CHANGED("Keen")
store.upsert("Keen", "character", store.get("Keen")["values"], picture=Image.new("RGB", (3, 3)))
check("a picture swapped in Manage re-runs the node instead of replaying the cache",
      pn.CardPresets.IS_CHANGED("Keen") != before)

check.done()
