"""Keep README.md's index in step with the nodes the pack actually registers.

    python tools/gen_readme_index.py            # rewrite the generated regions
    python tools/gen_readme_index.py --check    # verify only, non-zero exit if stale
    python tools/gen_readme_index.py --run-tests    # also refresh the tests badge

Run it with ComfyUI's own interpreter — it imports the pack, so it needs torch and comfy.

Only the regions between the `<!-- BEGIN GENERATED … -->` / `<!-- END GENERATED … -->` markers
are touched; every other line of the README is hand-written and left alone.

Two things are read rather than restated, so they cannot drift:

* the nodes — from ``NODE_CLASS_MAPPINGS`` / ``NODE_DISPLAY_NAME_MAPPINGS``;
* the doc layout — from ``docs/*.md`` themselves. A group's position comes from its
  ``<!-- index-order: N -->`` line, and a package section owns the folder named in its heading
  (``## … `local_llm/` — …``) unless it declares otherwise with ``<!-- packages: a, b -->``.

So adding a node needs no edit here at all, and adding a package needs only its `##` heading in
the right doc file. `--check` fails loudly when a node's folder has no section to live in, which
is the case a silent generator would paper over.

Beyond the index it audits the prose, because that is what actually goes stale:

* every registered node id must appear somewhere in the docs;
* every local link and anchor must resolve, with no duplicate anchors;
* every `backticked_snake_case` term must be a real node input or output — anything else has to
  be listed in ``tools/known_terms.txt`` (sampler names, llama.cpp flags, comfy internals …).
  Rename an input and this is what notices the README still uses the old name;
* every ``CATEGORY`` must be one of the paths declared in ``categories.py``, no suite may be split
  across two menu folders (nor let a stranger move into its folder), and every ``Kinburg-Nodes/…``
  path quoted in the docs must be a path that exists. Re-file a node and this is what notices the
  docs still send people to the old folder.
"""
import argparse
import importlib.util
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
COMFY = os.path.dirname(os.path.dirname(PACK))
DOCS = os.path.join(PACK, 'docs')
README = os.path.join(PACK, 'README.md')
KNOWN = os.path.join(HERE, 'known_terms.txt')

BADGE_ID, INDEX_ID = 'badges', 'index'


# --------------------------------------------------------------------------- pack
def load_pack():
    """Import the whole pack. The folder name has a hyphen in it, so a plain import can't work."""
    for p in (COMFY, os.path.join(COMFY, 'custom_nodes')):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        'kinburg_nodes', os.path.join(PACK, '__init__.py'), submodule_search_locations=[PACK])
    mod = importlib.util.module_from_spec(spec)
    sys.modules['kinburg_nodes'] = mod
    spec.loader.exec_module(mod)
    return mod


def read(path):
    """Text mode on purpose: universal newlines fold CRLF to \\n, so every comparison in here is
    newline-agnostic. `core.autocrlf` is on in this repo, so a fresh checkout really is CRLF."""
    return io.open(path, encoding='utf-8').read()


def newline_of(path):
    """Whatever the file already uses, so rewriting it doesn't flip every line ending."""
    blob = io.open(path, 'rb').read()
    return '\r\n' if blob.count(b'\r\n') > blob.count(b'\n') - blob.count(b'\r\n') else '\n'


def slug(heading):
    """GitHub's heading-anchor algorithm: lowercase, drop punctuation and symbols (emoji
    included), spaces become hyphens. This is why the emoji in a heading leaves a leading `-`."""
    t = re.sub(r'^#+\s*', '', heading).strip()
    t = re.sub(r'<[^>]+>', '', t).replace('&amp;', '&').lower()
    return re.sub(r'\s', '-', re.sub(r'[^a-z0-9_\s-]', '', t))


# --------------------------------------------------------------------------- docs layout
def read_docs():
    """[{file, title, order, packages:[{heading, anchor, label, folders}]}], in index order."""
    groups = []
    for name in sorted(os.listdir(DOCS)):
        if not name.endswith('.md'):
            continue
        lines = read(os.path.join(DOCS, name)).split('\n')
        if not lines[0].startswith('# '):
            raise SystemExit('%s: first line must be the group title (`# …`)' % name)
        m = re.search(r'<!--\s*index-order:\s*(\d+)\s*-->', '\n'.join(lines[:6]))
        if not m:
            raise SystemExit('%s: add `<!-- index-order: N -->` near the top so the index knows '
                             'where this group goes' % name)
        packages = []
        for i, line in enumerate(lines):
            if not line.startswith('## '):
                continue
            decl = re.search(r'<!--\s*packages:\s*([^>]+?)\s*-->', '\n'.join(lines[i + 1:i + 4]))
            if decl:
                folders = [f.strip() for f in decl.group(1).split(',') if f.strip()]
            else:
                folder = re.search(r'`([a-z0-9_]+)/`', line)
                if not folder:
                    raise SystemExit(
                        '%s: heading %r names no `folder/` — add one, or declare the folders it '
                        'documents with `<!-- packages: … -->`' % (name, line.strip()))
                folders = [folder.group(1)]
            packages.append({'heading': line[3:].strip(), 'anchor': slug(line),
                             'label': re.sub(r'^\S+\s+', '', line[3:].strip()),
                             'folders': folders})
        groups.append({'file': name, 'title': lines[0][2:].strip(),
                       'order': int(m.group(1)), 'packages': packages})
    groups.sort(key=lambda g: g['order'])
    seen = {}
    for g in groups:
        if g['order'] in seen:
            raise SystemExit('docs/%s and docs/%s share index-order %d'
                             % (seen[g['order']], g['file'], g['order']))
        seen[g['order']] = g['file']
    return groups


# --------------------------------------------------------------------------- generated text
def build_badges(version, node_count, checks):
    return '\n'.join([
        '[![version](https://img.shields.io/badge/version-%s-blue.svg)](pyproject.toml)' % version,
        '[![nodes](https://img.shields.io/badge/nodes-%d-orange.svg)](#-node-index)' % node_count,
        '[![tests](https://img.shields.io/badge/tests-%d%%20checks-brightgreen.svg)](#-tests)'
        % checks,
        '[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)',
        '[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)',
        '[![ComfyUI Manager](https://img.shields.io/badge/ComfyUI--Manager-installable-8A2BE2.svg)]'
        '(https://github.com/ltdrdata/ComfyUI-Manager)',
    ])


def build_index(groups, classes, names, by_folder):
    out = ['## 📍 Node Index', '',
           'All **%d** nodes, grouped by the package they live in. Every package links to its '
           'full documentation under [`docs/`](docs).' % len(classes), '']
    for g in groups:
        out += ['### %s' % g['title'], '',
                '📖 **[%s](docs/%s)**' % (g['title'], g['file']), '',
                '| Package | Nodes |', '|---|---|']
        for p in g['packages']:
            ids = sorted((n for f in p['folders'] for n in by_folder.get(f, [])),
                         key=lambda n: names.get(n, n))
            listed = ', '.join('`%s`' % names.get(n, n) for n in ids) \
                or '— *no nodes of its own*'
            out.append('| [%s](docs/%s#%s) | %s |' % (p['label'], g['file'], p['anchor'], listed))
        out.append('')
    out += ['---', '', '<details>',
            '<summary><b>🔎 All node ids (for workflow JSON / bug reports)</b></summary>', '',
            '| Display name | Node id | Category |', '|---|---|---|']
    for nid in sorted(classes, key=lambda n: names.get(n, n)):
        out.append('| %s | `%s` | `%s` |'
                   % (names.get(nid, nid), nid, getattr(classes[nid], 'CATEGORY', '?')))
    out += ['', '</details>']
    return '\n'.join(out)


def splice(text, marker_id, body):
    begin = re.search(r'^<!--\s*BEGIN GENERATED %s\b[^>]*-->$' % marker_id, text, re.M)
    end = re.search(r'^<!--\s*END GENERATED %s\s*-->$' % marker_id, text, re.M)
    if not begin or not end or end.start() < begin.end():
        raise SystemExit('README.md: the BEGIN/END GENERATED %s markers are missing or crossed'
                         % marker_id)
    return text[:begin.end()] + '\n' + body + '\n' + text[end.start():]


# --------------------------------------------------------------------------- audits
def audit(groups, classes, names, by_folder, check):
    docs = {'docs/' + g['file']: read(os.path.join(DOCS, g['file'])) for g in groups}
    docs['README.md'] = read(README)

    # 1. every node's folder has exactly one section willing to document it
    claimed = {}
    for g in groups:
        for p in g['packages']:
            for f in p['folders']:
                claimed.setdefault(f, []).append('docs/%s → %s' % (g['file'], p['label']))
    dupes = {f: w for f, w in claimed.items() if len(w) > 1}
    check('no package folder is claimed by two sections', not dupes, dupes or '')
    orphans = sorted({cls.__module__.split('.')[1] for cls in classes.values()}
                     - set(claimed))
    check('every node folder has a docs section', not orphans,
          ('undocumented: ' + ', '.join(orphans)) if orphans else '')

    # 2. every registered node is actually written about somewhere
    prose = '\n'.join(docs.values())
    unmentioned = sorted(n for n in classes if n not in prose and names.get(n, n) not in prose)
    check('every node is mentioned in the docs', not unmentioned,
          ('missing: ' + ', '.join(unmentioned)) if unmentioned else '')

    # 3. links and anchors
    anchors, dup = {}, []
    for path, text in docs.items():
        seen = {}
        for line in text.split('\n'):
            if re.match(r'^#{1,6} ', line):
                a = slug(line)
                seen[a] = seen.get(a, 0) + 1
        anchors[path] = seen
        dup += ['%s#%s' % (path, a) for a, n in seen.items() if n > 1]
    check('no duplicate heading anchors', not dup, ', '.join(dup))

    broken, total = [], 0
    for path, text in docs.items():
        base = os.path.dirname(path)
        for target in re.findall(r'\]\(([^)\s]+)\)', text):
            if target.startswith(('http://', 'https://', 'mailto:')):
                continue
            total += 1
            rel, _, frag = target.partition('#')
            key = path
            if rel:
                disk = os.path.normpath(os.path.join(PACK, base, rel))
                if not os.path.exists(disk):
                    broken.append('%s → %s (no such file)' % (path, target))
                    continue
                key = os.path.relpath(disk, PACK).replace(os.sep, '/')
            if frag and frag not in anchors.get(key, {}):
                broken.append('%s → %s (no such anchor)' % (path, target))
    check('all %d local links resolve' % total, not broken, '; '.join(broken))

    # 4. backticked parameter names still exist on some node
    real = set()
    for cls in classes.values():
        try:
            spec = cls.INPUT_TYPES()
        except Exception:
            spec = {}
        for section in ('required', 'optional', 'hidden'):
            real |= set((spec.get(section) or {}).keys())
        real |= set(getattr(cls, 'RETURN_NAMES', ()) or ())
    known = {ln.split('#')[0].strip() for ln in read(KNOWN).split('\n')} if os.path.exists(KNOWN) \
        else set()
    tokens = set(re.findall(r'`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`', prose))
    stale = sorted(tokens - real - known - {''})
    check('every `snake_case` term is a real input/output or a known term', not stale,
          ('unknown: ' + ', '.join(stale) + '  → fix the docs, or add it to '
           'tools/known_terms.txt') if stale else '')

    # 5. the menu tree matches categories.py, and no suite has leaked out of its own folder
    cats = importlib.import_module('kinburg_nodes.categories')
    used = {getattr(cls, 'CATEGORY', '?') for cls in classes.values()}
    undeclared = sorted(used - set(cats.ALL))
    check('every CATEGORY is declared in categories.py', not undeclared,
          ('undeclared: ' + ', '.join(undeclared) + '  → use a constant, or add one') if undeclared
          else '')

    owner_of = {home: folder for folder, home in cats.SUITES.items()}
    strays = []
    for nid, cls in sorted(classes.items()):
        folder, cat = cls.__module__.split('.')[1], getattr(cls, 'CATEGORY', '?')
        home = cats.SUITES.get(folder)
        # a suite's nodes belong in its own folder, or on the shared shelf at the bestiary root
        if home and cat not in (home, cats.CAT_BESTIARY):
            strays.append('%s lives in %s/ but shows up under %s' % (nid, folder, cat))
        # and nothing else may move in with them
        if cat in owner_of and owner_of[cat] != folder:
            strays.append('%s lives in %s/ but shows up in the %s suite'
                          % (nid, folder, owner_of[cat]))
    check('no suite is split across categories', not strays, '; '.join(strays))

    # 6. the docs quote category paths in prose — those must be paths that exist
    quoted = set(re.findall(r'`(Kinburg-Nodes/[A-Za-z/]*)`', prose))
    wrong = sorted(quoted - set(cats.ALL))
    check('every category path quoted in the docs is real', not wrong,
          ('stale: ' + ', '.join(wrong)) if wrong else '')
    return sorted(tokens - real - {''})


def test_checks():
    """The real check count, straight from the suite, for the badge."""
    py = sys.executable
    for cand in (os.path.join(COMFY, '.venv', 'Scripts', 'python.exe'),
                 os.path.join(COMFY, '.venv', 'bin', 'python')):
        if os.path.isfile(cand):
            py = cand
            break
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    p = subprocess.run([py, os.path.join(PACK, 'tests', 'run.py')], cwd=COMFY, env=env,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    m = re.search(r'^(\d+) checks in \d+ suite', p.stdout or '', re.M)
    if p.returncode or not m:
        raise SystemExit('tests did not pass, so the badge was not refreshed:\n'
                         + (p.stdout or '') + (p.stderr or ''))
    return int(m.group(1))


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='verify only; non-zero exit if the README is stale or an audit fails')
    ap.add_argument('--run-tests', action='store_true',
                    help='run tests/run.py and refresh the tests badge from its real count')
    args = ap.parse_args()

    fails = []

    def check(label, cond, extra=''):
        print(('  ok   ' if cond else '  FAIL ') + label + (('  ' + str(extra)) if extra else ''))
        if not cond:
            fails.append(label)

    pack = load_pack()
    classes, names = pack.NODE_CLASS_MAPPINGS, pack.NODE_DISPLAY_NAME_MAPPINGS
    by_folder = {}
    for nid, cls in classes.items():
        by_folder.setdefault(cls.__module__.split('.')[1], []).append(nid)

    groups = read_docs()
    current = read(README)

    version = re.search(r'^version\s*=\s*"([^"]+)"',
                        read(os.path.join(PACK, 'pyproject.toml')), re.M).group(1)
    if args.run_tests:
        checks = test_checks()
    else:
        found = re.search(r'badge/tests-(\d+)%20checks', current)
        checks = int(found.group(1)) if found else 0

    wanted = splice(current, BADGE_ID, build_badges(version, len(classes), checks))
    wanted = splice(wanted, INDEX_ID, build_index(groups, classes, names, by_folder))

    audit(groups, classes, names, by_folder, check)

    if args.check:
        check('README.md index is up to date', wanted == current,
              '' if wanted == current else 'run: python tools/gen_readme_index.py')
    elif wanted != current:
        io.open(README, 'w', encoding='utf-8', newline=newline_of(README)).write(wanted)
        print('  ok   README.md index rewritten')
    else:
        print('  ok   README.md index already up to date')

    print('\n%d nodes, %d packages in %d docs files'
          % (len(classes), sum(len(g['packages']) for g in groups), len(groups)))
    if fails:
        print('FAILED: ' + ', '.join(fails))
        return 1
    print('ALL PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
