import os
import time
import json
import sqlite3
import re
import numpy as np
from collections import defaultdict, Counter
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import config

def clean_str(s: str) -> str:
    """Normalize string for fuzzy comparison."""
    if not s:
        return ""
    return re.sub(r'[\W_]+', ' ', s).strip().lower()

def get_primary_artist(artist: str) -> str:
    """Extract primary artist name before collaborations / features."""
    if not artist:
        return ""
    parts = re.split(r'\s*(?:feat\.?|ft\.?|featuring|vs\.?|&|with|w/|,|/)\s*', artist, flags=re.I)
    return parts[0].strip().lower()

AUSTRO_GERMAN_ARTISTS = {
    'die jungen zillertaler', 'zillertaler schürzenjäger', 'original zilllertaler',
    'andreas gabalier', 'dj ötzi', 'dj ötzi & nik p.', 'helene fischer', 'edmund',
    'seiler und speer', 'seiler und speer feat. anton horvath', 'pizzera & jaus',
    'erste allgemeine verunsicherung', 'rainhard fendrich', 'wolfgang ambros',
    'wolfgang ambros & andré heller', 'ambros/tauchen/prokopetz', 'sts',
    'hubert von goisern', 'georg danzer', 'die seer', 'die steirische landstreich',
    'die landstreich', 'die doofen', 'die toten hosen', 'die ärzte',
    'ostbahn kurti & die chefpartie', 'kurt pascher und seine böhmerwälder musikanten',
    'karl hanspeter & seine original südtirol musikanten', 'original tiroler kaiserjägermusik',
    'militärmusik kärnten', 'die 3 tenöre', 'das modul', 'mickie krause', 'peter alexander', 'falco',
    'ludwig hirsch', 'steinbäcker', 'schiffkowitz', 'timischl', 'klostertaler',
    'kastelruther spatzen', 'schürzenjäger', 'die sonnwendler', 'zillertaler heimatbuben', 'austria 3'
}

def artist_name_token_similarity(a1: str, a2: str) -> float:
    """Matches distinctive regional / brand tokens (e.g. zillertal, ambros, speer, gabalier)."""
    if not a1 or not a2:
        return 0.0
    stop = {'the', 'die', 'der', 'das', 'and', 'und', '&', 'feat', 'ft', 'featuring', 'various', 'artists', 'de', 'la', 'von'}
    a1_norm = re.sub(r'(.)\1+', r'\1\1', a1.lower())
    a2_norm = re.sub(r'(.)\1+', r'\1\1', a2.lower())
    t1 = set(re.findall(r'\b\w{4,}\b', a1_norm)) - stop
    t2 = set(re.findall(r'\b\w{4,}\b', a2_norm)) - stop
    for w1 in t1:
        for w2 in t2:
            if w1 == w2 or (len(w1) >= 6 and len(w2) >= 6 and (w1 in w2 or w2 in w1)):
                return 1.0
    return 0.0

def get_macro_cluster(track: dict) -> str:
    """Classify track into macro cultural and musical clusters."""
    art = (track.get('artist') or '').lower().strip()
    g = (track.get('genre') or '').lower().strip()
    
    # 1. Classical / Orchestral
    if g in ['classical', 'baroque', 'opera', 'orchestral'] or any(k in art for k in ['orchestra', 'philharmonic', 'symphon', 'beethoven', 'mozart', 'bach', 'chopin', 'schubert', 'vivaldi', 'karajan', 'harnoncourt', 'camerata']):
        return 'classical'
        
    # 2. Austro-German / Volksmusik / Schlager / Austropop
    art_norm = re.sub(r'(.)\1+', r'\1\1', art)
    if art in AUSTRO_GERMAN_ARTISTS or any(k in art_norm for k in ['zillert', 'österreich', 'austria', 'steirisch', 'tirol', 'schürzenjäger', 'gabalier', 'fendrich', 'ambros', 'ötzi', 'spatzen', 'volksmusik']):
        return 'volksmusik_schlager'
    if g in ['volksmusik', 'german folk', 'german pop', 'schlager', 'austropop']:
        if 'creedence' not in art:
            return 'volksmusik_schlager'

    # 3. Hip-Hop / Rap
    if g in ['hip-hop', 'hip-hop/rap', 'rap & hip-hop', 'rap', 'trap']:
        return 'hiphop'
        
    # 4. Electronic / Dance
    if g in ['dance', 'dance & house', 'dance & dj', 'club', 'trance', 'electronica', 'edm']:
        return 'electronic'
        
    # 5. Country & Americana
    if g in ['country', 'country & folk', 'americana', 'bluegrass']:
        return 'country'
        
    # 6. Rock & Metal
    if g in ['rock', 'hard rock', 'metal', 'heavy', 'classic rock', 'alternative rock', 'grunge', 'indie rock']:
        return 'rock'
        
    return 'pop'

class SessionContext:
    def __init__(self, seed_id: str):
        self.seed_id = seed_id
        self.history = [seed_id]
        self.artist_history = []
        self.last_activity = time.time()

    def update(self, track_id: str, artist: str):
        self.history.append(track_id)
        if artist:
            self.artist_history.append(artist)
        self.last_activity = time.time()
        if len(self.history) > 50:
            self.history.pop(0)
        if len(self.artist_history) > 50:
            self.artist_history.pop(0)

class AutoplayEngine:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or self._resolve_db_path()
        self.features_db_path = "/root/navidrome-autoplay/features.db"
        self.tracks = {}              # id -> dict
        self.track_ids = []           # list of ids
        self.id_to_idx = {}           # id -> matrix row index
        self.idx_to_id = {}           # matrix row index -> id
        self.artist_tracks = defaultdict(list)
        self.album_tracks = defaultdict(list)
        self.artist_name_to_id = {}
        self.artist_id_to_name = {}
        self.artist_similar_db = defaultdict(set)
        self.artist_cooccur = defaultdict(float)
        self.track_transitions = defaultdict(float)
        self.track_embeddings = None  # (N, D) normalized dense matrix
        self.sessions = {}            # session_id/seed_id -> SessionContext
        self.annotations = {}         # id -> {'play_count': int, 'starred': bool}
        
        self.load_database()

    def _resolve_db_path(self) -> str:
        if os.path.exists(config.PRIMARY_DB_PATH) and os.access(config.PRIMARY_DB_PATH, os.R_OK):
            return config.PRIMARY_DB_PATH
        if os.path.exists(config.BACKUP_DB_PATH):
            return config.BACKUP_DB_PATH
        raise FileNotFoundError(f"Navidrome database not found at {config.PRIMARY_DB_PATH} or {config.BACKUP_DB_PATH}")

    def load_database(self):
        print(f"[AutoplayEngine] Loading database from: {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # 1. Load Tracks
        self.tracks.clear()
        self.artist_tracks.clear()
        self.album_tracks.clear()

        rows = c.execute("""
            SELECT id, title, artist, album, artist_id, album_id, year, bpm, duration, path 
            FROM media_file
        """).fetchall()

        for r in rows:
            tid, title, artist, album, artist_id, album_id, year, bpm, duration, path = r
            self.tracks[tid] = {
                'id': tid,
                'title': title or 'Unknown Title',
                'artist': artist or 'Unknown Artist',
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
                'energy': 0.5,
                'cluster': 'pop',
                'clean_key': (clean_str(title), clean_str(artist))
            }
            if artist:
                self.artist_tracks[artist].append(tid)
                if artist_id:
                    self.artist_name_to_id[artist] = artist_id
                    self.artist_id_to_name[artist_id] = artist
            if album:
                self.album_tracks[album].append(tid)

        # 1b. Load Features DB (BPM, Camelot Key, Genre, Energy)
        if os.path.exists(self.features_db_path):
            try:
                f_conn = sqlite3.connect(self.features_db_path)
                fc = f_conn.cursor()
                f_rows = fc.execute("SELECT item_id, bpm, key_name, camelot, genre, energy FROM track_features").fetchall()
                for fid, fbpm, fkey, fcam, fgen, fen in f_rows:
                    if fid in self.tracks:
                        t = self.tracks[fid]
                        if fbpm > 0:
                            t['bpm'] = fbpm
                        if fkey:
                            t['key_name'] = fkey
                        if fcam:
                            t['camelot'] = fcam
                        if fgen:
                            t['genre'] = fgen
                        if fen > 0:
                            t['energy'] = fen
                f_conn.close()
                print(f"[AutoplayEngine] Enriched {len(f_rows)} tracks from features cache.")
            except Exception as e:
                print(f"[AutoplayEngine] Warning loading features cache: {e}")

        # Classify macro clusters
        for t in self.tracks.values():
            t['cluster'] = get_macro_cluster(t)

        self.track_ids = list(self.tracks.keys())
        self.id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.idx_to_id = {i: tid for i, tid in enumerate(self.track_ids)}
        n_tracks = len(self.track_ids)
        print(f"[AutoplayEngine] Loaded {n_tracks} tracks across {len(self.artist_tracks)} artists.")

        # 2. Load Annotations (play counts & starred)
        try:
            for r in c.execute("SELECT item_id, play_count, starred FROM annotation WHERE item_type='media_file'").fetchall():
                self.annotations[r[0]] = {
                    'play_count': r[1] or 0,
                    'starred': bool(r[2])
                }
        except Exception as e:
            print(f"[AutoplayEngine] Warning loading annotations: {e}")

        # 3. Load External / Similar Artists
        try:
            for aname, sim_json in c.execute("SELECT name, similar_artists FROM artist WHERE similar_artists != '[]' AND similar_artists IS NOT NULL").fetchall():
                try:
                    data = json.loads(sim_json)
                    for item in data:
                        target = item.get('name')
                        if target:
                            self.artist_similar_db[aname].add(target)
                            self.artist_cooccur[(aname, target)] += 4.0
                            self.artist_cooccur[(target, aname)] += 4.0
                except Exception:
                    pass
        except Exception as e:
            print(f"[AutoplayEngine] Warning loading similar artists: {e}")

        # 4. Multi-Signal Hybrid Transition Matrix
        print("[AutoplayEngine] Building dense multi-signal graph matrix...")
        self.track_transitions.clear()
        mat = dok_matrix((n_tracks, n_tracks), dtype=np.float32)

        # 4a. Album Co-Membership (Natural sequential flow)
        for album, tids in self.album_tracks.items():
            if len(tids) > 1:
                L = len(tids)
                for i in range(L):
                    idx_i = self.id_to_idx[tids[i]]
                    for j in range(i + 1, min(L, i + 12)):
                        idx_j = self.id_to_idx[tids[j]]
                        mat[idx_i, idx_j] += 1.5
                        mat[idx_j, idx_i] += 1.5

        # 4b. Same Artist Continuity
        for artist, tids in self.artist_tracks.items():
            if len(tids) > 1:
                L = len(tids)
                for i in range(L):
                    idx_i = self.id_to_idx[tids[i]]
                    for j in range(i + 1, min(L, i + 10)):
                        idx_j = self.id_to_idx[tids[j]]
                        mat[idx_i, idx_j] += 0.8
                        mat[idx_j, idx_i] += 0.8

        # 4c. Cultural Cluster & Token Nearest Neighbors (Solves 90% Cold-Start Gap)
        cluster_tids = defaultdict(list)
        for tid, t in self.tracks.items():
            cluster_tids[t['cluster']].append(tid)

        for clus, tids in cluster_tids.items():
            sorted_tids = sorted(tids, key=lambda x: self.tracks[x]['bpm'])
            L = len(sorted_tids)
            for i in range(L):
                idx_i = self.id_to_idx[sorted_tids[i]]
                cand_i = self.tracks[sorted_tids[i]]
                for j in range(max(0, i - 2), min(L, i + 3)):
                    if i != j:
                        idx_j = self.id_to_idx[sorted_tids[j]]
                        cand_j = self.tracks[sorted_tids[j]]
                        w = 0.5
                        if artist_name_token_similarity(cand_i['artist'], cand_j['artist']) > 0:
                            w += 2.0
                        mat[idx_i, idx_j] += w

        # 4d. User Scrobbles & Playlists (Ground truth listening history)
        try:
            scrobbles = c.execute("SELECT user_id, media_file_id, submission_time FROM scrobbles ORDER BY user_id, submission_time ASC").fetchall()
            curr_s = []
            last_u, last_t = None, None
            for u, tid, t_sub in scrobbles:
                if tid not in self.id_to_idx:
                    continue
                if last_u != u or (last_t is not None and (t_sub - last_t) > 1500):
                    if len(curr_s) >= 2:
                        L = len(curr_s)
                        for i in range(L):
                            for j in range(max(0, i - 4), min(L, i + 5)):
                                if i != j:
                                    u_idx = self.id_to_idx[curr_s[i]]
                                    v_idx = self.id_to_idx[curr_s[j]]
                                    w = 1.0 / abs(i - j)
                                    mat[u_idx, v_idx] += w
                                    self.track_transitions[(u_idx, v_idx)] += w
                    curr_s = [tid]
                else:
                    curr_s.append(tid)
                last_u, last_t = u, t_sub
        except Exception as e:
            print(f"[AutoplayEngine] Warning loading scrobbles: {e}")

        try:
            for (pid,) in c.execute("SELECT id FROM playlist").fetchall():
                pts = [r[0] for r in c.execute("SELECT media_file_id FROM playlist_tracks WHERE playlist_id=? ORDER BY id ASC", (pid,)).fetchall() if r[0] in self.id_to_idx]
                if len(pts) >= 2:
                    L = len(pts)
                    for i in range(L):
                        for j in range(max(0, i - 4), min(L, i + 5)):
                            if i != j:
                                u_idx = self.id_to_idx[pts[i]]
                                v_idx = self.id_to_idx[pts[j]]
                                w = 1.2 / abs(i - j)
                                mat[u_idx, v_idx] += w
                                self.track_transitions[(u_idx, v_idx)] += w
        except Exception as e:
            print(f"[AutoplayEngine] Warning loading playlists: {e}")

        # 5. SVD Latent Graph Embedding (Dense representation across 100% of tracks)
        print("[AutoplayEngine] Computing latent graph embeddings via TruncatedSVD...")
        csr_mat = mat.tocsr()
        csr_mat = csr_mat + csr_matrix(np.eye(n_tracks, dtype=np.float32) * 0.1)
        
        n_components = min(32, n_tracks - 1)
        if n_components > 2 and csr_mat.nnz > 10:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            dense_emb = svd.fit_transform(csr_mat)
            self.track_embeddings = normalize(dense_emb, norm='l2', axis=1)
        else:
            self.track_embeddings = np.zeros((n_tracks, 16), dtype=np.float32)

        conn.close()
        print("[AutoplayEngine] Engine initialization complete and ready.")

    def reload_features(self):
        if not os.path.exists(self.features_db_path):
            return 0
        try:
            f_conn = sqlite3.connect(self.features_db_path)
            fc = f_conn.cursor()
            f_rows = fc.execute("SELECT item_id, bpm, key_name, camelot, genre, energy FROM track_features").fetchall()
            for fid, fbpm, fkey, fcam, fgen, fen in f_rows:
                if fid in self.tracks:
                    t = self.tracks[fid]
                    if fbpm > 0:
                        t['bpm'] = fbpm
                    if fkey:
                        t['key_name'] = fkey
                    if fcam:
                        t['camelot'] = fcam
                    if fgen:
                        t['genre'] = fgen
                    if fen > 0:
                        t['energy'] = fen
            f_conn.close()
            # Update macro clusters
            for t in self.tracks.values():
                t['cluster'] = get_macro_cluster(t)
            return len(f_rows)
        except Exception as e:
            print(f"[AutoplayEngine] Error reloading features: {e}")
            return 0

    def _get_or_create_session(self, session_key: str, seed_id: str) -> SessionContext:
        now = time.time()
        expired = [k for k, v in self.sessions.items() if (now - v.last_activity) > config.SESSION_TTL_SECONDS]
        for k in expired:
            del self.sessions[k]

        if session_key not in self.sessions:
            self.sessions[session_key] = SessionContext(seed_id)
        return self.sessions[session_key]

    def _camelot_distance(self, key_a: str, key_b: str) -> float:
        """Calculates distance on the Camelot wheel (e.g. '8A' to '8B' = 0.5, '8A' to '9A' = 1.0)."""
        if not key_a or not key_b:
            return -1.0
        if key_a == key_b:
            return 0.0
        try:
            num_a, mode_a = int(key_a[:-1]), key_a[-1]
            num_b, mode_b = int(key_b[:-1]), key_b[-1]
            wheel_dist = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
            mode_dist = 0.0 if mode_a == mode_b else 0.5
            return wheel_dist + mode_dist
        except Exception:
            return -1.0

    def _bpm_penalty(self, bpm_curr: float, bpm_cand: float) -> float:
        """Octave-folded tempo transition penalty (tolerance for halftime / doubletime)."""
        if bpm_curr <= 0 or bpm_cand <= 0:
            return 0.0
        diffs = [
            abs(bpm_curr - bpm_cand),
            abs(bpm_curr - bpm_cand * 2.0),
            abs(bpm_curr * 2.0 - bpm_cand)
        ]
        pct = min(diffs) / bpm_curr
        if pct <= config.BPM_TOLERANCE_PCT:
            return 0.0
        return min(0.35, float((pct - config.BPM_TOLERANCE_PCT) * 1.5))

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
        if item_id not in self.tracks:
            return []

        sess_key = session_id or item_id
        session = self._get_or_create_session(sess_key, item_id)
        session.update(item_id, self.tracks[item_id]['artist'])

        macro_seed_id = session.seed_id
        macro_idx = self.id_to_idx.get(macro_seed_id, self.id_to_idx[item_id])
        curr_idx = self.id_to_idx[item_id]

        curr_track = self.tracks[item_id]
        curr_art = curr_track['artist']
        curr_folder = curr_track['path'].split('/')[0] if '/' in curr_track['path'] else ''
        curr_bpm = curr_track['bpm']
        curr_year = curr_track['year']
        curr_cluster = curr_track.get('cluster', 'pop')

        macro_track = self.tracks[macro_seed_id]
        macro_art = macro_track['artist']
        macro_year = macro_track['year']
        macro_cluster = macro_track.get('cluster', 'pop')

        # 1. SVD Latent Graph Vector Proximities
        if self.track_embeddings is not None:
            curr_vec = self.track_embeddings[curr_idx]
            macro_vec = self.track_embeddings[macro_idx]
            sim_local_svd = np.dot(self.track_embeddings, curr_vec)
            sim_macro_svd = np.dot(self.track_embeddings, macro_vec)
        else:
            sim_local_svd = np.zeros(len(self.track_ids))
            sim_macro_svd = np.zeros(len(self.track_ids))

        # 2. Score Candidates
        scores = {}
        for idx, tid in enumerate(self.track_ids):
            if tid == item_id:
                continue

            cand = self.tracks[tid]
            cand_art = cand['artist']
            cand_cluster = cand.get('cluster', 'pop')

            # Base score from Latent SVD space
            score = (
                config.LOCAL_TRANSITION_WEIGHT * float(sim_local_svd[idx]) +
                config.MACRO_ANCHOR_WEIGHT * float(sim_macro_svd[idx])
            )

            # 2a. Cultural & Macro-Cluster Guardrails
            if cand_cluster == curr_cluster:
                score += 0.55
            elif curr_cluster == 'classical' or cand_cluster == 'classical':
                score -= 2.50 # Never mix classical with pop/rock/volksmusik
            elif curr_cluster == 'volksmusik_schlager' and cand_cluster != 'volksmusik_schlager':
                score -= 1.60 # Keep Volksmusik/Schlager sessions pure
            elif cand_cluster == 'volksmusik_schlager' and curr_cluster != 'volksmusik_schlager':
                score -= 1.60
            elif curr_cluster == 'rock' and cand_cluster == 'electronic':
                score -= 0.50

            # 2b. Direct Transition Boost (from scrobbles/playlists)
            if (curr_idx, idx) in self.track_transitions:
                score += min(2.5, self.track_transitions[(curr_idx, idx)] * 0.4)

            # 2c. Artist-Level Transition & Token Affinity
            tok_sim = artist_name_token_similarity(curr_art, cand_art)
            if tok_sim > 0:
                score += 1.0 # Strong affinity for related regional bands (e.g. Zillertaler)

            if cand_art == curr_art:
                score += 0.8
                if cand['album'] == curr_track['album']:
                    score += 0.4
            elif (curr_art, cand_art) in self.artist_cooccur:
                co_w = self.artist_cooccur[(curr_art, cand_art)]
                score += min(2.0, co_w * 0.25)
            elif (macro_art, cand_art) in self.artist_cooccur:
                co_w = self.artist_cooccur[(macro_art, cand_art)]
                score += min(1.2, co_w * 0.15)

            # 2d. Era / Decade Matching
            if curr_year > 0 and cand['year'] > 0:
                y_diff = abs(curr_year - cand['year'])
                if y_diff <= 6:
                    score += 0.35 * (1.0 - y_diff / 6.0)

            # 2e. Compilation / Folder affinity
            cand_folder = cand['path'].split('/')[0] if '/' in cand['path'] else ''
            if curr_folder and cand_folder == curr_folder:
                score += 0.25

            # 2f. User Affinity Boost
            ann = self.annotations.get(tid)
            if ann:
                if ann['starred']:
                    score += 0.35
                if ann['play_count'] >= 5:
                    score += 0.20

            # 2g. Specific Genre Matching Boost
            curr_genre = curr_track.get('genre', '').lower()
            cand_genre = cand.get('genre', '').lower()
            if curr_genre and cand_genre:
                if curr_genre == cand_genre or curr_genre in cand_genre or cand_genre in curr_genre:
                    score += 0.40

            # 2h. Harmonic Key Mixing (Camelot Wheel)
            curr_cam = curr_track.get('camelot', '')
            cand_cam = cand.get('camelot', '')
            if curr_cam and cand_cam:
                cam_dist = self._camelot_distance(curr_cam, cand_cam)
                if cam_dist == 0.0 or cam_dist == 0.5:
                    score += 0.20
                elif cam_dist == 1.0:
                    score += 0.12
                elif cam_dist >= 3.0:
                    score -= min(0.25, (cam_dist - 2.0) * 0.08)

            # 2i. BPM Transition Continuity (Octave-folded)
            if curr_bpm > 0 and cand['bpm'] > 0:
                ratio = cand['bpm'] / curr_bpm
                folded = min(abs(ratio - 1.0), abs(ratio - 0.5), abs(ratio - 2.0))
                if folded <= config.BPM_TOLERANCE_PCT:
                    score += 0.15
                else:
                    score -= min(0.40, folded * 0.80)

            # 2j. Energy Dynamic Continuity
            curr_en = curr_track.get('energy', 0.5)
            cand_en = cand.get('energy', 0.5)
            en_diff = abs(curr_en - cand_en)
            if en_diff > 0.25:
                score -= (en_diff - 0.25) * 0.40

            scores[tid] = score

        # 3. Stochastic Sampling / Controlled Serendipity (Spotify BaRT Style)
        # Apply Gumbel-Top-k perturbation so repeated Instant Mixes on the same track yield fresh, diverse tracklists
        if config.RANDOMIZATION_TEMPERATURE > 0:
            tau = config.RANDOMIZATION_TEMPERATURE
            max_score = max(scores.values()) if scores else 0.0
            min_viable = max(0.15, max_score * 0.45)
            
            perturbed_scores = {}
            for tid, s in scores.items():
                if s >= min_viable:
                    u = np.random.uniform(1e-10, 1.0 - 1e-10)
                    g = -np.log(-np.log(u))
                    perturbed_scores[tid] = s + tau * g
                else:
                    perturbed_scores[tid] = s
            ranked = sorted(perturbed_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Apply Fatigue, Deduplication, and Diversity Filters
        results = []
        artist_counts = Counter()
        seen_keys = set()
        recent_played = set(session.history[-8:])

        seen_keys.add(curr_track['clean_key'])

        for tid, score in ranked:
            if tid in recent_played:
                continue

            cand = self.tracks[tid]
            cand_art = cand['artist']
            cand_key = cand['clean_key']

            # Deduplication across CD releases
            if eliminate_duplicates and cand_key in seen_keys:
                continue

            # Primary artist fatigue cooldown
            prim_art = get_primary_artist(cand_art)
            if artist_counts[prim_art] >= config.MAX_ARTIST_REPEAT_IN_QUEUE:
                continue

            # Additional penalty if artist played immediately prior
            if session.artist_history and prim_art == get_primary_artist(session.artist_history[-1]):
                if len(self.artist_tracks) > 5 and artist_counts[prim_art] >= 1:
                    continue

            artist_counts[prim_art] += 1
            seen_keys.add(cand_key)

            # Distance = normalized inverse score for Navidrome PDK
            norm_dist = max(0.01, min(0.99, float(1.0 / (1.0 + max(0.0, score)))))

            results.append({
                "item_id": cand['id'],
                "title": cand['title'],
                "author": cand['artist'],
                "album": cand['album'],
                "distance": round(norm_dist, 4)
            })

            if len(results) >= n:
                break

        return results

    def get_similar_artists(self, artist_identifier: str, count: int = 10) -> list[dict]:
        """Find similar artists by name or ID using hybrid graph transitions."""
        artist_name = artist_identifier
        if artist_identifier in self.artist_id_to_name:
            artist_name = self.artist_id_to_name[artist_identifier]

        artist_scores = defaultdict(float)

        # 1. Co-occurrence edges
        for (a1, a2), w in self.artist_cooccur.items():
            if a1 == artist_name and a2 in self.artist_tracks:
                artist_scores[a2] += w

        # 2. Similar artists from database
        if artist_name in self.artist_similar_db:
            for related in self.artist_similar_db[artist_name]:
                if related in self.artist_tracks:
                    artist_scores[related] += 5.0

        # 3. Token similarity
        for cand_art in self.artist_tracks.keys():
            if cand_art != artist_name and artist_name_token_similarity(artist_name, cand_art) > 0:
                artist_scores[cand_art] += 8.0

        ranked = sorted(artist_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for aname, _ in ranked[:count]:
            aid = self.artist_name_to_id.get(aname, "")
            results.append({
                "artist": aname,
                "artist_id": aid,
                "component_matches": []
            })
        return results

    def find_path(self, start_song_id: str, end_song_id: str, count: int = 5, max_steps: int = 10) -> dict:
        """Finds the smoothest musical DJ path between Song A and Song B."""
        if start_song_id not in self.tracks or end_song_id not in self.tracks:
            return {"path": []}

        import heapq

        start_idx = self.id_to_idx[start_song_id]
        end_idx = self.id_to_idx[end_song_id]

        queue = [(0.0, [start_idx])]
        visited = set()

        best_path = None

        while queue:
            cost, path = heapq.heappop(queue)
            curr = path[-1]

            if curr == end_idx:
                best_path = path
                break

            if curr in visited or len(path) > max_steps:
                continue
            visited.add(curr)

            curr_tid = self.idx_to_id[curr]
            neighbors = self.get_similar_tracks(curr_tid, n=8, eliminate_duplicates=True)

            for nb in neighbors:
                nb_idx = self.id_to_idx[nb['item_id']]
                if nb_idx not in visited:
                    edge_cost = nb['distance']
                    heapq.heappush(queue, (cost + edge_cost, path + [nb_idx]))

        if not best_path:
            best_path = [start_idx, end_idx]

        path_tracks = []
        for idx in best_path:
            t = self.tracks[self.idx_to_id[idx]]
            path_tracks.append({
                "item_id": t['id'],
                "title": t['title'],
                "author": t['artist'],
                "album": t['album'],
                "distance": 0.0
            })

        return {"path": path_tracks}
