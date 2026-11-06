import os
import re
import copy
import json
import time
import threading
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import coo_matrix, identity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import config

# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def clean_str(s: str) -> str:
    """Normalize string for fuzzy comparison."""
    if not s:
        return ""
    return re.sub(r'[\W_]+', ' ', s).strip().lower()


# Collaboration separators. Keywords need word boundaries so that names like
# "Daft Punk", "Taylor Swift", "Within Temptation" or "Bill Withers" stay intact.
# '&' and '/' without spaces are NOT split ("Simon & Garfunkel", "AC/DC" are band names).
_COLLAB_RE = re.compile(
    r'\s*[\(\[]?\b(?:feat\.?|ft\.?|featuring|vs\.?|with)(?=\s|$)'
    r'|\s+w/\s+'
    r'|\s*[,;]\s*'
    r'|\s+/\s+',
    flags=re.I,
)


def get_primary_artist(artist: str) -> str:
    """Extract primary artist name before collaborations / features."""
    if not artist:
        return ""
    primary = _COLLAB_RE.split(artist, maxsplit=1)[0].strip().lower()
    return primary or artist.strip().lower()


AUSTRO_GERMAN_ARTISTS = {
    'die jungen zillertaler', 'zillertaler schürzenjäger', 'original zilllertaler',
    'andreas gabalier', 'dj ötzi', 'dj ötzi & nik p.', 'helene fischer', 'edmund',
    'seiler und speer', 'seiler und speer feat. anton horvath', 'pizzera & jaus',
    'erste allgemeine verunsicherung', 'rainhard fendrich', 'wolfgang ambros',
    'wolfgang ambros & andré heller', 'ambros/tauchen/prokopetz', 'sts',
    'hubert von goisern', 'georg danzer', 'die seer', 'die steirische landstreich',
    'die landstreich', 'die doofen',
    'ostbahn kurti & die chefpartie', 'kurt pascher und seine böhmerwälder musikanten',
    'karl hanspeter & seine original südtirol musikanten', 'original tiroler kaiserjägermusik',
    'militärmusik kärnten', 'die 3 tenöre', 'das modul', 'mickie krause', 'peter alexander', 'falco',
    'ludwig hirsch', 'steinbäcker', 'schiffkowitz', 'timischl', 'klostertaler',
    'kastelruther spatzen', 'schürzenjäger', 'die sonnwendler', 'zillertaler heimatbuben', 'austria 3'
}

_AUSTRO_RE = re.compile(
    r'\b(?:zillert\w*|österreich\w*|austria|steirisch\w*|tirol\w*|schürzenjäger|'
    r'gabalier|fendrich|ambros|ötzi|spatzen|volksmusik\w*)\b'
)

# Only unambiguous classical markers. Plain 'orchestra' / 'symphon' / 'bach' substrings
# wrongly caught Electric Light Orchestra, OMD, Symphony X, Bachman-Turner Overdrive, Bacharach.
_CLASSICAL_ARTIST_RE = re.compile(
    r'\b(?:philharmoni\w*|symphony orchestra|chamber orchestra|sinfonieorchester|'
    r'symphonieorchester|kammerorchester|staatskapelle|beethoven|mozart|'
    r'(?:j\.\s?s\.|johann sebastian) bach|chopin|schubert|vivaldi|karajan|harnoncourt|camerata)\b'
)

_GENRE_TO_CLUSTER = {}
for _c, _gs in {
    'classical': ['classical', 'baroque', 'opera', 'orchestral', 'klassik'],
    'volksmusik_schlager': ['volksmusik', 'german folk', 'german pop', 'schlager', 'austropop'],
    'hiphop': ['hip-hop', 'hip hop', 'hip-hop/rap', 'rap & hip-hop', 'rap', 'trap'],
    'electronic': ['dance', 'dance & house', 'dance & dj', 'club', 'trance', 'electronica', 'edm',
                   'electronic', 'house', 'techno'],
    'country': ['country', 'country & folk', 'americana', 'bluegrass'],
    'rock': ['rock', 'hard rock', 'metal', 'heavy', 'heavy metal', 'classic rock', 'alternative rock',
             'grunge', 'indie rock', 'punk', 'punk rock', 'deutschrock', 'alternative'],
}.items():
    for _g in _gs:
        _GENRE_TO_CLUSTER[_g] = _c


def _genre_cluster(g: str):
    if not g:
        return None
    if g in _GENRE_TO_CLUSTER:
        return _GENRE_TO_CLUSTER[g]
    first = re.split(r'\s*[;,|/]\s*', g)[0]
    return _GENRE_TO_CLUSTER.get(first)


def get_macro_cluster(track: dict) -> str:
    """Classify track into macro cultural and musical clusters."""
    art = (track.get('artist') or '').lower().strip()
    g = (track.get('genre') or '').lower().strip()
    g_cluster = _genre_cluster(g)

    # 1. Classical by genre tag
    if g_cluster == 'classical':
        return 'classical'

    # 2. Austro-German / Volksmusik / Schlager / Austropop (artist list wins over generic tags)
    art_norm = re.sub(r'(.)\1+', r'\1\1', art)
    if art in AUSTRO_GERMAN_ARTISTS or _AUSTRO_RE.search(art_norm):
        return 'volksmusik_schlager'

    # 3. Any other known genre tag
    if g_cluster:
        return g_cluster

    # 4. No usable genre: fall back to unambiguous classical artist markers
    if _CLASSICAL_ARTIST_RE.search(art):
        return 'classical'

    return 'pop'


# Words that are too common in artist names to imply any relation between two artists.
_TOKEN_STOPWORDS = {
    'the', 'die', 'der', 'das', 'and', 'und', 'feat', 'featuring', 'various', 'artists', 'von',
    'band', 'boys', 'girls', 'brothers', 'sisters', 'music', 'musik', 'musikanten', 'orchestra',
    'orchester', 'black', 'white', 'love', 'little', 'king', 'kings', 'queen', 'young', 'jungen',
    'original', 'live', 'group', 'project', 'club', 'sound', 'sounds', 'stars', 'kids', 'family',
    'friends', 'singers', 'choir', 'trio', 'quartet', 'quartett', 'ensemble', 'blues', 'soul',
    'street', 'city', 'night', 'dead', 'death', 'electric', 'light', 'heart', 'their', 'seine',
    'with', 'mitglieder', 'chor', 'company', 'gang', 'crew', 'experience', 'unknown', 'artist',
    # common first names ("Michael Jackson" is not related to "George Michael")
    'michael', 'john', 'johnny', 'paul', 'peter', 'george', 'david', 'james', 'robert', 'richard',
    'thomas', 'mark', 'chris', 'frank', 'tony', 'andreas', 'wolfgang', 'hans', 'karl', 'kurt',
    'georg', 'ludwig', 'helene', 'mickie', 'steve', 'billy', 'bill', 'jack', 'mike', 'dave',
    'marc', 'marco', 'anna', 'maria', 'christian', 'stefan', 'martin', 'alexander', 'daniel',
}


def artist_tokens(name: str) -> set:
    """Distinctive tokens of an artist name (lowercased, repeated letters collapsed)."""
    if not name:
        return set()
    norm = re.sub(r'(.)\1+', r'\1\1', name.lower())
    return set(re.findall(r'\b\w{4,}\b', norm)) - _TOKEN_STOPWORDS


def artist_name_token_similarity(a1: str, a2: str) -> float:
    """Matches distinctive regional / brand tokens (e.g. zillertal, ambros, speer, gabalier).

    Stand-alone version without library statistics (the engine additionally ignores tokens
    that occur in many artist names of the library).
    """
    t1, t2 = artist_tokens(a1), artist_tokens(a2)
    for w1 in t1:
        for w2 in t2:
            if w1 == w2 or (len(w1) >= 6 and len(w2) >= 6 and (w1.startswith(w2) or w2.startswith(w1))):
                return 1.0
    return 0.0


def parse_camelot(cam: str):
    """'8A' -> (8, 0), '12B' -> (12, 1). Returns (0, 0) if unknown."""
    if not cam:
        return 0, 0
    m = re.fullmatch(r'0?(\d{1,2})([AB])', cam.strip().upper())
    if not m or not 1 <= int(m.group(1)) <= 12:
        return 0, 0
    return int(m.group(1)), (1 if m.group(2) == 'B' else 0)


CLUSTERS = ['pop', 'classical', 'volksmusik_schlager', 'hiphop', 'electronic', 'country', 'rock']
_C = {name: i for i, name in enumerate(CLUSTERS)}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionContext:
    MAX_REMEMBERED_RECS = 500

    def __init__(self, seed_id: str, seed_artist: str = ""):
        self.seed_id = seed_id
        self.history = [seed_id]
        self.artist_history = [seed_artist] if seed_artist else []
        self.recent_recs = []
        self.last_activity = time.time()

    def update(self, track_id: str, artist: str):
        self.history.append(track_id)
        if artist:
            self.artist_history.append(artist)
        self.last_activity = time.time()
        del self.history[:-50]
        del self.artist_history[:-50]

    def remember_recs(self, ids):
        self.recent_recs.extend(ids)
        del self.recent_recs[:-self.MAX_REMEMBERED_RECS]


# ---------------------------------------------------------------------------
# Immutable-after-build engine state (swapped atomically on reload)
# ---------------------------------------------------------------------------

class EngineState:
    def __init__(self):
        self.db_path = ""
        self.tracks = {}
        self.track_ids = []
        self.id_to_idx = {}
        self.artist_tracks = defaultdict(list)
        self.artist_names = []          # artist index -> name
        self.artist_index = {}          # name -> artist index
        self.artist_name_to_id = {}
        self.artist_id_to_name = {}
        self.artist_related = {}        # artist index -> set of token-related artist indices
        self.artist_similar_db = defaultdict(set)
        self.artist_cooccur = defaultdict(dict)   # name -> {name: weight}
        self.annotations = {}
        self.transitions = None         # csr (n, n) direct transition weights
        self.embeddings = None          # (n, D) L2-normalized
        self.features_mtime = 0.0
        # per-track arrays (aligned with track_ids)
        self.a_artist = None
        self.a_album = None
        self.a_folder = None
        self.a_year = None
        self.a_starred = None
        self.a_playcount = None
        self.a_cluster = None
        self.a_bpm = None
        self.a_energy = None
        self.a_cam_num = None
        self.a_cam_mode = None
        self.a_genre = None
        self.genres = []
        self.prim_artist = []
        self.clean_keys = []


def _features_mtime(path: str) -> float:
    m = 0.0
    for p in (path, path + '-wal'):
        try:
            m = max(m, os.path.getmtime(p))
        except OSError:
            pass
    return m


def _apply_features(tracks: dict, features_db_path: str) -> int:
    """Overlay analyzer results onto track dicts. Energy becomes a 0..1 library percentile."""
    if not os.path.exists(features_db_path):
        return 0
    import sqlite3
    conn = sqlite3.connect(features_db_path, timeout=10)
    try:
        rows = conn.execute(
            "SELECT item_id, bpm, key_name, camelot, genre, energy FROM track_features").fetchall()
    finally:
        conn.close()

    raw_energy = {}
    for fid, fbpm, fkey, fcam, fgen, fen in rows:
        t = tracks.get(fid)
        if t is None:
            continue
        if fbpm and fbpm > 0:
            t['bpm'] = fbpm
        if fkey:
            t['key_name'] = fkey
        if fcam:
            t['camelot'] = fcam
        if fgen:
            t['genre'] = fgen
        # Older analyzer versions stored a 0.5 placeholder when no audio was decoded.
        if fen is not None and fen > 0 and abs(fen - 0.5) > 1e-9:
            raw_energy[fid] = fen

    if raw_energy:
        ids = list(raw_energy)
        vals = np.array([raw_energy[i] for i in ids])
        ranks = vals.argsort().argsort() / max(1, len(vals) - 1)
        for tid, r in zip(ids, ranks):
            tracks[tid]['energy'] = float(r)
    return len(rows)


def _compute_feature_arrays(st: EngineState):
    """(Re)compute arrays that depend on analyzer features."""
    tr = [st.tracks[tid] for tid in st.track_ids]
    for t in tr:
        t['cluster'] = get_macro_cluster(t)
    st.a_cluster = np.array([_C[t['cluster']] for t in tr], dtype=np.int8)
    st.a_bpm = np.array([t['bpm'] or 0.0 for t in tr], dtype=np.float64)
    st.a_energy = np.array([np.nan if t['energy'] is None else t['energy'] for t in tr])
    cams = [parse_camelot(t['camelot']) for t in tr]
    st.a_cam_num = np.array([c[0] for c in cams], dtype=np.int16)
    st.a_cam_mode = np.array([c[1] for c in cams], dtype=np.int8)
    genre_index = {}
    gidx = []
    for t in tr:
        g = (t['genre'] or '').lower().strip()
        if not g:
            gidx.append(-1)
            continue
        if g not in genre_index:
            genre_index[g] = len(genre_index)
        gidx.append(genre_index[g])
    st.genres = list(genre_index)
    st.a_genre = np.array(gidx, dtype=np.int32)


def build_state(db_path: str, features_db_path: str) -> EngineState:
    st = EngineState()
    st.db_path = db_path
    conn = config.connect_ro(db_path)
    c = conn.cursor()
    try:
        # 1. Tracks, ordered so album windows follow the real track order
        rows = c.execute("""
            SELECT id, title, artist, album, artist_id, album_id, year, bpm, duration, path
            FROM media_file
            ORDER BY album_id, disc_number, track_number, path
        """).fetchall()

        album_tracks = defaultdict(list)
        for tid, title, artist, album, artist_id, album_id, year, bpm, duration, path in rows:
            artist_name = artist or 'Unknown Artist'
            st.tracks[tid] = {
                'id': tid,
                'title': title or 'Unknown Title',
                'artist': artist_name,
                'album': album or 'Unknown Album',
                'artist_id': artist_id or '',
                'album_id': album_id or '',
                'year': year or 0,
                'bpm': bpm or 0.0,
                'duration': duration or 0.0,
                'path': path or '',
                'genre': '',
                'key_name': '',
                'camelot': '',
                'energy': None,
                'cluster': 'pop',
                'clean_key': (clean_str(title), clean_str(artist)),
            }
            st.artist_tracks[artist_name].append(tid)
            if artist and artist_id:
                st.artist_name_to_id[artist] = artist_id
                st.artist_id_to_name[artist_id] = artist
            # Group by album_id: album *names* like "Greatest Hits" collide across artists
            album_tracks[album_id or f"{album}\x00{artist_name}"].append(tid)

        _apply_features(st.tracks, features_db_path)
        st.features_mtime = _features_mtime(features_db_path)

        st.track_ids = list(st.tracks.keys())
        st.id_to_idx = {tid: i for i, tid in enumerate(st.track_ids)}
        n = len(st.track_ids)
        print(f"[AutoplayEngine] Loaded {n} tracks across {len(st.artist_tracks)} artists.")

        # Static per-track arrays
        st.artist_names = list(st.artist_tracks.keys())
        st.artist_index = {a: i for i, a in enumerate(st.artist_names)}
        tr = [st.tracks[tid] for tid in st.track_ids]
        st.a_artist = np.array([st.artist_index[t['artist']] for t in tr], dtype=np.int32)
        album_keys = {}
        st.a_album = np.array([album_keys.setdefault(t['album_id'] or (t['album'], t['artist']), len(album_keys))
                               for t in tr], dtype=np.int32)
        st.a_year = np.array([t['year'] for t in tr], dtype=np.int32)
        # Folder affinity uses the parent directory, ignoring huge flat dump folders
        parents = [os.path.dirname(t['path']) for t in tr]
        pcount = Counter(parents)
        folder_keys = {}
        st.a_folder = np.array([folder_keys.setdefault(p, len(folder_keys)) if p and pcount[p] <= 300 else -1
                                for p in parents], dtype=np.int32)
        st.prim_artist = [get_primary_artist(t['artist']) for t in tr]
        st.clean_keys = [t['clean_key'] for t in tr]
        _compute_feature_arrays(st)

        # 2. Token-related artists (ignore tokens shared by many artists of the library)
        tokens = {a: artist_tokens(a) for a in st.artist_names}
        df = Counter(tok for ts in tokens.values() for tok in ts)
        max_df = max(5, int(0.01 * len(st.artist_names)))
        tok_index = defaultdict(set)
        for a, ts in tokens.items():
            for tok in ts:
                if df[tok] <= max_df:
                    tok_index[tok].add(st.artist_index[a])
        # prefix relations between long tokens ("zillertal" ~ "zillertaler")
        sorted_toks = sorted(t for t in tok_index if len(t) >= 6)
        tok_related = defaultdict(set)
        for i, t1 in enumerate(sorted_toks):
            for t2 in sorted_toks[i + 1:]:
                if not t2.startswith(t1):
                    break
                tok_related[t1].add(t2)
                tok_related[t2].add(t1)
        for a, ts in tokens.items():
            ai = st.artist_index[a]
            rel = set()
            for tok in ts:
                if tok in tok_index:
                    rel |= tok_index[tok]
                    for t2 in tok_related.get(tok, ()):
                        rel |= tok_index[t2]
            rel.discard(ai)
            if rel:
                st.artist_related[ai] = rel

        # 3. Annotations (aggregated across users)
        try:
            for item_id, pc, starred in c.execute(
                    "SELECT item_id, SUM(COALESCE(play_count, 0)), MAX(COALESCE(starred, 0)) "
                    "FROM annotation WHERE item_type='media_file' GROUP BY item_id"):
                st.annotations[item_id] = {'play_count': pc or 0, 'starred': bool(starred)}
        except Exception as e:
            print(f"[AutoplayEngine] Warning loading annotations: {e}")
        st.a_starred = np.array([st.annotations.get(tid, {}).get('starred', False) for tid in st.track_ids])
        st.a_playcount = np.array([st.annotations.get(tid, {}).get('play_count', 0) for tid in st.track_ids])

        # 4. External / similar artists
        try:
            for aname, sim_json in c.execute(
                    "SELECT name, similar_artists FROM artist "
                    "WHERE similar_artists IS NOT NULL AND similar_artists != '[]'"):
                try:
                    for item in json.loads(sim_json):
                        target = item.get('name')
                        if target and target != aname:
                            st.artist_similar_db[aname].add(target)
                            st.artist_cooccur[aname][target] = st.artist_cooccur[aname].get(target, 0.0) + 4.0
                            st.artist_cooccur[target][aname] = st.artist_cooccur[target].get(aname, 0.0) + 4.0
                except Exception:
                    pass
        except Exception as e:
            print(f"[AutoplayEngine] Warning loading similar artists: {e}")

        # 5. Multi-signal hybrid graph (built as COO triplets, duplicates are summed)
        print("[AutoplayEngine] Building multi-signal graph matrix...")
        rows_i, cols_i, vals = [], [], []
        t_rows, t_cols, t_vals = [], [], []
        idx = st.id_to_idx

        def add_window(tids, window, weight, symmetric=True):
            L = len(tids)
            for i in range(L):
                a = idx[tids[i]]
                for j in range(i + 1, min(L, i + window)):
                    b = idx[tids[j]]
                    rows_i.append(a); cols_i.append(b); vals.append(weight)
                    if symmetric:
                        rows_i.append(b); cols_i.append(a); vals.append(weight)

        # 5a. Album co-membership (natural sequential flow)
        for tids in album_tracks.values():
            add_window(tids, 12, 1.5)
        # 5b. Same artist continuity
        for tids in st.artist_tracks.values():
            add_window(tids, 10, 0.8)
        # 5c. Cluster BPM neighbours + token-related artists (only tracks with a known BPM)
        cluster_tids = defaultdict(list)
        for tid in st.track_ids:
            if st.tracks[tid]['bpm'] > 0:
                cluster_tids[st.tracks[tid]['cluster']].append(tid)
        for tids in cluster_tids.values():
            tids.sort(key=lambda x: st.tracks[x]['bpm'])
            L = len(tids)
            for i in range(L):
                a = idx[tids[i]]
                ai = st.a_artist[a]
                for j in range(max(0, i - 2), min(L, i + 3)):
                    if i != j:
                        b = idx[tids[j]]
                        w = 0.5
                        if st.a_artist[b] in st.artist_related.get(ai, ()):
                            w += 2.0
                        rows_i.append(a); cols_i.append(b); vals.append(w)

        def add_sequence(seq, base_w):
            L = len(seq)
            for i in range(L):
                a = idx[seq[i]]
                for j in range(max(0, i - 4), min(L, i + 5)):
                    if i != j:
                        b = idx[seq[j]]
                        w = base_w / abs(i - j)
                        rows_i.append(a); cols_i.append(b); vals.append(w)
                        t_rows.append(a); t_cols.append(b); t_vals.append(w)
            # artist-level transitions between consecutive tracks
            for x, y in zip(seq, seq[1:]):
                ax, ay = st.tracks[x]['artist'], st.tracks[y]['artist']
                if ax != ay:
                    st.artist_cooccur[ax][ay] = st.artist_cooccur[ax].get(ay, 0.0) + 1.0
                    st.artist_cooccur[ay][ax] = st.artist_cooccur[ay].get(ax, 0.0) + 1.0

        # 5d. Scrobble listening sessions (ground truth)
        try:
            curr_s, last_u, last_t = [], None, None
            for u, tid, t_sub in c.execute(
                    "SELECT user_id, media_file_id, submission_time FROM scrobbles "
                    "ORDER BY user_id, submission_time ASC"):
                if tid not in idx:
                    continue
                if last_u != u or (last_t is not None and (t_sub - last_t) > 1500):
                    if len(curr_s) >= 2:
                        add_sequence(curr_s, 1.0)
                    curr_s = [tid]
                else:
                    curr_s.append(tid)
                last_u, last_t = u, t_sub
            if len(curr_s) >= 2:  # flush the final session
                add_sequence(curr_s, 1.0)
        except Exception as e:
            print(f"[AutoplayEngine] Warning loading scrobbles: {e}")

        # 5e. Playlists
        try:
            playlists = defaultdict(list)
            for pid, mfid in c.execute(
                    "SELECT playlist_id, media_file_id FROM playlist_tracks ORDER BY playlist_id, id"):
                if mfid in idx:
                    playlists[pid].append(mfid)
            for pts in playlists.values():
                if len(pts) >= 2:
                    add_sequence(pts, 1.2)
        except Exception as e:
            print(f"[AutoplayEngine] Warning loading playlists: {e}")
    finally:
        conn.close()

    shape = (n, n)
    mat = coo_matrix((np.array(vals, dtype=np.float32), (rows_i, cols_i)), shape=shape).tocsr()
    mat = mat + identity(n, dtype=np.float32, format='csr') * 0.1
    st.transitions = coo_matrix((np.array(t_vals, dtype=np.float32), (t_rows, t_cols)), shape=shape).tocsr()

    # 6. SVD latent graph embedding (dense representation across 100% of tracks)
    print("[AutoplayEngine] Computing latent graph embeddings via TruncatedSVD...")
    n_components = min(32, n - 1)
    if n_components > 2 and mat.nnz > 10:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        st.embeddings = normalize(svd.fit_transform(mat), norm='l2', axis=1)
    else:
        st.embeddings = np.zeros((n, 16), dtype=np.float32)
    return st


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AutoplayEngine:
    def __init__(self, db_path: str = None, features_db_path: str = None):
        self.db_path = db_path
        self.features_db_path = features_db_path or config.FEATURES_DB_PATH
        self.state = EngineState()
        self.sessions = {}
        self._sess_lock = threading.Lock()
        self._reload_lock = threading.Lock()

        if not self.db_path:
            try:
                self.db_path = config.resolve_db_path()
            except FileNotFoundError as e:
                print(f"[AutoplayEngine] Warning: {e}. Waiting for database configuration.")
                return

        self.load_database()

    # -- compatibility accessors (read the current snapshot) --
    @property
    def tracks(self):
        return self.state.tracks

    @property
    def artist_tracks(self):
        return self.state.artist_tracks

    @property
    def artist_cooccur(self):
        return self.state.artist_cooccur

    def artist_connection_count(self) -> int:
        return sum(len(v) for v in self.state.artist_cooccur.values()) // 2

    # -- loading --
    def load_database(self, blocking: bool = True) -> bool:
        """Rebuild everything off to the side, then swap it in atomically.
        Returns False if another reload is already running and blocking=False."""
        if not self._reload_lock.acquire(blocking=blocking):
            return False
        try:
            if not self.db_path:
                self.db_path = config.resolve_db_path()
            print(f"[AutoplayEngine] Loading database from: {self.db_path}")
            new_state = build_state(self.db_path, self.features_db_path)
            self.state = new_state  # atomic reference swap
            print("[AutoplayEngine] Engine initialization complete and ready.")
            return True
        finally:
            self._reload_lock.release()

    def reload_features(self) -> int:
        """Pick up new analyzer results (only when features.db changed). Returns rows read."""
        mtime = _features_mtime(self.features_db_path)
        if not mtime or mtime <= self.state.features_mtime:
            return 0
        if not self._reload_lock.acquire(blocking=False):
            return 0
        try:
            old = self.state
            st = copy.copy(old)
            st.tracks = {tid: dict(t) for tid, t in old.tracks.items()}
            count = _apply_features(st.tracks, self.features_db_path)
            st.features_mtime = mtime
            _compute_feature_arrays(st)
            self.state = st
            return count
        except Exception as e:
            print(f"[AutoplayEngine] Error reloading features: {e}")
            return 0
        finally:
            self._reload_lock.release()

    # -- scoring --
    @staticmethod
    def _camelot_distance(key_a: str, key_b: str) -> float:
        """Distance on the Camelot wheel (e.g. '8A' to '8B' = 0.5, '8A' to '9A' = 1.0)."""
        na, ma = parse_camelot(key_a)
        nb, mb = parse_camelot(key_b)
        if not na or not nb:
            return -1.0
        wheel = min(abs(na - nb), 12 - abs(na - nb))
        return wheel + (0.0 if ma == mb else 0.5)

    def _score(self, st: EngineState, curr_idx: int, macro_idx: int) -> np.ndarray:
        """Vectorised hybrid score of every track as successor of curr_idx (anchored at macro_idx)."""
        n = len(st.track_ids)
        emb = st.embeddings
        score = (config.LOCAL_TRANSITION_WEIGHT * (emb @ emb[curr_idx]) +
                 config.MACRO_ANCHOR_WEIGHT * (emb @ emb[macro_idx])).astype(np.float64)

        # a. Cultural & macro-cluster guardrails
        cc = st.a_cluster[curr_idx]
        cand = st.a_cluster
        same = cand == cc
        score[same] += 0.55
        rest = ~same
        m_cls = rest & ((cc == _C['classical']) | (cand == _C['classical']))
        score[m_cls] -= 2.50
        rest &= ~m_cls
        m_vo = rest & ((cc == _C['volksmusik_schlager']) | (cand == _C['volksmusik_schlager']))
        score[m_vo] -= 1.60
        rest &= ~m_vo
        if cc == _C['rock']:
            score[rest & (cand == _C['electronic'])] -= 0.50

        # b. Direct transitions (scrobbles / playlists)
        row = st.transitions.getrow(curr_idx)
        if row.nnz:
            score[row.indices] += np.minimum(2.5, row.data * 0.4)

        # c. Artist affinity
        ca = st.a_artist[curr_idx]
        related = st.artist_related.get(ca)
        if related:
            score[np.isin(st.a_artist, list(related))] += 1.0
        same_art = st.a_artist == ca
        score[same_art] += 0.8
        score[same_art & (st.a_album == st.a_album[curr_idx])] += 0.4

        curr_art = st.artist_names[ca]
        macro_art = st.artist_names[st.a_artist[macro_idx]]
        art_bonus = np.zeros(len(st.artist_names))
        for name, w in st.artist_cooccur.get(macro_art, {}).items():
            ai = st.artist_index.get(name)
            if ai is not None:
                art_bonus[ai] = min(1.2, w * 0.15)
        for name, w in st.artist_cooccur.get(curr_art, {}).items():
            ai = st.artist_index.get(name)
            if ai is not None:
                art_bonus[ai] = min(2.0, w * 0.25)
        score += np.where(same_art, 0.0, art_bonus[st.a_artist])

        # d. Era matching
        cy = st.a_year[curr_idx]
        if cy > 0:
            d = np.abs(st.a_year - cy)
            score += np.where((st.a_year > 0) & (d <= 6), 0.35 * (1.0 - d / 6.0), 0.0)

        # e. Folder affinity
        cf = st.a_folder[curr_idx]
        if cf >= 0:
            score[st.a_folder == cf] += 0.25

        # f. User affinity
        score[st.a_starred] += 0.35
        score[st.a_playcount >= 5] += 0.20

        # g. Genre match
        cg = st.a_genre[curr_idx]
        if cg >= 0:
            g0 = st.genres[cg]
            gmatch = np.array([g0 == g or g0 in g or g in g0 for g in st.genres] + [False])
            score[gmatch[st.a_genre]] += 0.40  # index -1 hits the trailing False

        # h. Harmonic mixing (Camelot wheel)
        cn = st.a_cam_num[curr_idx]
        if cn > 0:
            known = st.a_cam_num > 0
            diff = np.abs(st.a_cam_num.astype(np.int32) - cn)
            dist = np.minimum(diff, 12 - diff) + np.where(st.a_cam_mode == st.a_cam_mode[curr_idx], 0.0, 0.5)
            score[known & (dist <= 0.5)] += 0.20
            score[known & (dist == 1.0)] += 0.12
            far = known & (dist >= 3.0)
            score[far] -= np.minimum(0.25, (dist[far] - 2.0) * 0.08)

        # i. BPM continuity (octave-folded)
        cb = st.a_bpm[curr_idx]
        if cb > 0:
            known = st.a_bpm > 0
            ratio = st.a_bpm / cb
            folded = np.minimum(np.abs(ratio - 1.0), np.minimum(np.abs(ratio - 0.5), np.abs(ratio - 2.0)))
            ok = known & (folded <= config.BPM_TOLERANCE_PCT)
            bad = known & ~ok
            score[ok] += 0.15
            score[bad] -= np.minimum(0.40, folded[bad] * 0.80)

        # j. Energy continuity (percentiles; unknown energy = no penalty)
        ce = st.a_energy[curr_idx]
        if not np.isnan(ce):
            ed = np.abs(st.a_energy - ce)
            pen = np.where(ed > 0.25, (ed - 0.25) * 0.40, 0.0)
            score -= np.nan_to_num(pen, nan=0.0)

        score[curr_idx] = -np.inf
        return score

    # -- sessions --
    def _session_snapshot(self, st: EngineState, session_id, item_id):
        key = session_id or item_id
        artist = st.tracks[item_id]['artist']
        now = time.time()
        with self._sess_lock:
            for k in [k for k, v in self.sessions.items() if (now - v.last_activity) > config.SESSION_TTL_SECONDS]:
                del self.sessions[k]
            sess = self.sessions.get(key)
            # A track we never played or recommended in this session means the user started a new mix.
            if sess is None or (item_id not in sess.history and item_id not in sess.recent_recs):
                sess = SessionContext(item_id, artist)
                self.sessions[key] = sess
            elif sess.history[-1] != item_id:
                sess.update(item_id, artist)
            else:
                sess.last_activity = now
            return sess, sess.seed_id, set(sess.history[-8:]), (sess.artist_history[-1] if sess.artist_history else "")

    def get_similar_tracks(
        self,
        item_id: str,
        n: int = 10,
        eliminate_duplicates: bool = True,
        session_id: str = None
    ) -> list[dict]:
        """
        Generate Autoplay / Instant Mix recommendations using Apple & Spotify hybrid logic:
        1. Macro-Anchor (prevents genre drift)
        2. Micro-Anchor (immediate transition flow)
        3. Dense Latent Graph Proximity (100% library coverage)
        4. Cultural & Macro-Cluster Guardrails (no genre clashes)
        5. Artist Token & Brand Continuity
        6. Camelot Wheel Harmonic Mixing
        7. BPM Pacing
        8. Primary Artist Fatigue & Cross-Compilation Deduplication
        """
        st = self.state  # one consistent snapshot for the whole request
        if item_id not in st.id_to_idx or n <= 0:
            return []
        n = min(n, config.MAX_RESULTS)

        sess, seed_id, recent_played, last_artist = self._session_snapshot(st, session_id, item_id)
        curr_idx = st.id_to_idx[item_id]
        macro_idx = st.id_to_idx.get(seed_id, curr_idx)

        scores = self._score(st, curr_idx, macro_idx)

        # Stochastic sampling / controlled serendipity (Gumbel-Top-k on viable candidates)
        if config.RANDOMIZATION_TEMPERATURE > 0:
            finite = np.isfinite(scores)
            max_score = scores[finite].max() if finite.any() else 0.0
            min_viable = max(0.15, max_score * 0.45)
            viable = finite & (scores >= min_viable)
            u = np.random.uniform(1e-10, 1.0 - 1e-10, size=len(scores))
            ranked_scores = scores + np.where(viable, config.RANDOMIZATION_TEMPERATURE * -np.log(-np.log(u)), 0.0)
        else:
            ranked_scores = scores

        # Fatigue, deduplication and diversity filters
        results = []
        artist_counts = Counter()
        seen_keys = {st.clean_keys[curr_idx]}
        last_prim = get_primary_artist(last_artist)
        many_artists = len(st.artist_names) > 5

        for idx in np.argsort(-ranked_scores):
            if not np.isfinite(scores[idx]):
                continue
            tid = st.track_ids[idx]
            if tid in recent_played:
                continue
            cand_key = st.clean_keys[idx]
            if eliminate_duplicates and cand_key in seen_keys:
                continue
            prim = st.prim_artist[idx]
            if artist_counts[prim] >= config.MAX_ARTIST_REPEAT_IN_QUEUE:
                continue
            if many_artists and prim == last_prim and artist_counts[prim] >= 1:
                continue

            artist_counts[prim] += 1
            seen_keys.add(cand_key)
            cand = st.tracks[tid]
            score = float(ranked_scores[idx])
            results.append({
                "item_id": tid,
                "title": cand['title'],
                "author": cand['artist'],
                "album": cand['album'],
                "distance": round(max(0.01, min(0.99, 1.0 / (1.0 + max(0.0, score)))), 4),
            })
            if len(results) >= n:
                break

        with self._sess_lock:
            sess.remember_recs(r['item_id'] for r in results)
        return results

    def get_similar_artists(self, artist_identifier: str, count: int = 10) -> list[dict]:
        """Find similar artists by name or ID using hybrid graph transitions."""
        st = self.state
        count = max(0, min(count, config.MAX_RESULTS))
        artist_name = st.artist_id_to_name.get(artist_identifier, artist_identifier)

        artist_scores = defaultdict(float)
        for a2, w in st.artist_cooccur.get(artist_name, {}).items():
            if a2 in st.artist_tracks:
                artist_scores[a2] += w
        for related in st.artist_similar_db.get(artist_name, ()):
            if related in st.artist_tracks:
                artist_scores[related] += 5.0
        ai = st.artist_index.get(artist_name)
        if ai is not None:
            for rel in st.artist_related.get(ai, ()):
                artist_scores[st.artist_names[rel]] += 3.0
        artist_scores.pop(artist_name, None)

        ranked = sorted(artist_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"artist": a, "artist_id": st.artist_name_to_id.get(a, ""), "component_matches": []}
                for a, _ in ranked[:count]]

    def find_path(self, start_song_id: str, end_song_id: str, count: int = 5, max_steps: int = 10) -> dict:
        """Smooth musical path from song A to song B.

        Deterministic greedy beam walk: from the current song, look at its best successors and
        take the one that is both a good transition and closer to the target in embedding space.
        Does not create sessions and costs one scoring pass per step.
        """
        st = self.state
        if start_song_id not in st.id_to_idx or end_song_id not in st.id_to_idx:
            return {"path": []}
        max_steps = max(2, min(max_steps, config.MAX_PATH_STEPS))

        emb = st.embeddings
        start_idx, end_idx = st.id_to_idx[start_song_id], st.id_to_idx[end_song_id]
        to_end = emb @ emb[end_idx]
        path, dists = [start_idx], [0.0]
        used_keys = {st.clean_keys[start_idx], st.clean_keys[end_idx]}
        curr = start_idx

        while curr != end_idx and len(path) < max_steps - 1:
            scores = self._score(st, curr, start_idx)
            top = np.argpartition(-scores, min(12, len(scores) - 1))[:12]
            if end_idx in top:
                break
            best, best_val = None, -np.inf
            for cand in top:
                if not np.isfinite(scores[cand]) or st.clean_keys[cand] in used_keys:
                    continue
                if to_end[cand] <= to_end[curr] - 0.05:  # must not move away from the target
                    continue
                val = scores[cand] + 2.0 * to_end[cand]
                if val > best_val:
                    best, best_val = cand, val
            if best is None:
                break
            path.append(int(best))
            dists.append(round(1.0 / (1.0 + max(0.0, float(scores[best]))), 4))
            used_keys.add(st.clean_keys[best])
            curr = best

        if path[-1] != end_idx:
            end_score = self._score(st, curr, start_idx)[end_idx]
            path.append(end_idx)
            dists.append(round(1.0 / (1.0 + max(0.0, float(end_score))), 4))

        out = []
        for idx, d in zip(path, dists):
            t = st.tracks[st.track_ids[idx]]
            out.append({"item_id": t['id'], "title": t['title'], "author": t['artist'],
                        "album": t['album'], "distance": d})
        return {"path": out}
