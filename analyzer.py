import os
import sys
import time
import fcntl
import re
import sqlite3
import subprocess
import multiprocessing as mp
import numpy as np
import scipy.signal
import aubio
import mutagen
import config

MUSIC_DIR = config.MUSIC_DIR
FEATURES_DB_PATH = config.FEATURES_DB_PATH
LOCK_PATH = FEATURES_DB_PATH + ".lock"
RETRY_FAILED_AFTER_SECONDS = 24 * 3600

CAMELOT_MAP = {
    'B major': '1B', 'G# minor': '1A', 'Ab minor': '1A',
    'F# major': '2B', 'Gb major': '2B', 'D# minor': '2A', 'Eb minor': '2A',
    'Db major': '3B', 'C# major': '3B', 'Bb minor': '3A', 'A# minor': '3A',
    'Ab major': '4B', 'G# major': '4B', 'F minor': '4A',
    'Eb major': '5B', 'D# major': '5B', 'C minor': '5A',
    'Bb major': '6B', 'A# major': '6B', 'G minor': '6A',
    'F major': '7B', 'D minor': '7A',
    'C major': '8B', 'A minor': '8A',
    'G major': '9B', 'E minor': '9A',
    'D major': '10B', 'B minor': '10A',
    'A major': '11B', 'F# minor': '11A', 'Gb minor': '11A',
    'E major': '12B', 'C# minor': '12A', 'Db minor': '12A',
}

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def init_features_db():
    conn = sqlite3.connect(FEATURES_DB_PATH, timeout=30)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS track_features (
            item_id TEXT PRIMARY KEY,
            bpm REAL,
            key_name TEXT,
            camelot TEXT,
            genre TEXT,
            energy REAL,
            updated_at REAL
        )
    """)
    cols = [r[1] for r in c.execute("PRAGMA table_info(track_features)")]
    if 'status' not in cols:
        # 'ok' | 'missing' (file not found) | 'error' (could not decode)
        c.execute("ALTER TABLE track_features ADD COLUMN status TEXT DEFAULT 'ok'")
    conn.commit()
    conn.close()


def parse_key_tag(raw: str):
    """Normalize key tags like 'Am', 'F#m', 'C# minor', 'Bbmaj', '8A', '08B' -> (key_name, camelot)."""
    raw = (raw or '').strip()
    if not raw:
        return "", ""
    m = re.fullmatch(r'0?(\d{1,2})\s*([ABab])', raw)
    if m and 1 <= int(m.group(1)) <= 12:
        cam = f"{int(m.group(1))}{m.group(2).upper()}"
        key = next((k for k, v in CAMELOT_MAP.items() if v == cam), "")
        return key, cam
    m = re.fullmatch(r'([A-Ga-g])\s*([#♯b♭]?)\s*(major|maj|minor|min|m|moll|dur)?', raw)
    if not m:
        return "", ""
    note = m.group(1).upper() + {'♯': '#', '♭': 'b'}.get(m.group(2), m.group(2))
    mode = (m.group(3) or '').lower()
    key = f"{note} {'minor' if mode in ('m', 'min', 'minor', 'moll') else 'major'}"
    return (key, CAMELOT_MAP[key]) if key in CAMELOT_MAP else ("", "")


def extract_key_from_pcm(samples, sr=22050):
    if len(samples) < sr * 5:
        return "", ""
    try:
        f, t, Zxx = scipy.signal.stft(samples, fs=sr, nperseg=2048, noverlap=1024)
        mag = np.abs(Zxx)

        valid_idx = (f >= 80) & (f <= 3000)
        f_valid = f[valid_idx]
        mag_valid = mag[valid_idx, :]

        midi = 12.0 * np.log2(f_valid / 440.0) + 69.0
        pitch_class = np.round(midi).astype(int) % 12

        chroma = np.zeros(12)
        for pc in range(12):
            chroma[pc] = np.sum(mag_valid[pitch_class == pc, :])

        norm_factor = np.linalg.norm(chroma - np.mean(chroma))
        if norm_factor == 0:
            return "", ""
        c_norm = (chroma - np.mean(chroma)) / norm_factor

        maj_p = (MAJOR_PROFILE - np.mean(MAJOR_PROFILE)) / np.linalg.norm(MAJOR_PROFILE - np.mean(MAJOR_PROFILE))
        min_p = (MINOR_PROFILE - np.mean(MINOR_PROFILE)) / np.linalg.norm(MINOR_PROFILE - np.mean(MINOR_PROFILE))

        best_corr = -2.0
        best_key = ""

        for i in range(12):
            c_m = np.dot(c_norm, np.roll(maj_p, i))
            if c_m > best_corr:
                best_corr = c_m
                best_key = f"{PITCH_NAMES[i]} major"
            c_mi = np.dot(c_norm, np.roll(min_p, i))
            if c_mi > best_corr:
                best_corr = c_mi
                best_key = f"{PITCH_NAMES[i]} minor"

        camelot = CAMELOT_MAP.get(best_key, "")
        return best_key, camelot
    except Exception:
        return "", ""

def extract_bpm_aubio(samples, sr=22050):
    try:
        win_s = 512
        hop_s = win_s // 2
        o = aubio.tempo('default', win_s, hop_s, sr)

        total_frames = len(samples)
        for pos in range(0, total_frames - hop_s, hop_s):
            frame = samples[pos:pos + hop_s]
            o(frame)
        bpm = float(o.get_bpm())
        if 40.0 <= bpm <= 240.0:
            return round(bpm, 1)
        return 0.0
    except Exception:
        return 0.0


def _decode_slice(full_path: str):
    """Decode 40s from offset 25s as mono float32 @ 22.05 kHz. Kills ffmpeg on timeout."""
    cmd = ['ffmpeg', '-ss', '25', '-t', '40', '-i', full_path,
           '-f', 'f32le', '-ac', '1', '-ar', '22050', '-v', 'error', '-']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        raw, _ = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return None
    samples = np.frombuffer(raw, dtype=np.float32)
    if len(samples) < 22050 * 5:
        # Short track: retry from the beginning
        proc = subprocess.Popen(['ffmpeg', '-t', '40', '-i', full_path, '-f', 'f32le', '-ac', '1',
                                 '-ar', '22050', '-v', 'error', '-'],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            raw, _ = proc.communicate(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            return None
        samples = np.frombuffer(raw, dtype=np.float32)
    return samples if len(samples) >= 22050 * 5 else None


def analyze_track(item):
    """Returns a row for track_features. Never returns None, so failures are recorded
    (with a status) instead of being retried on every run."""
    item_id, rel_path, db_bpm, db_genre = item
    full_path = os.path.join(MUSIC_DIR, rel_path or '')
    now = time.time()

    if not rel_path or not os.path.exists(full_path):
        return (item_id, 0.0, "", "", db_genre or "", None, now, 'missing')

    genre = db_genre or ""
    bpm = float(db_bpm) if db_bpm else 0.0
    key_name, camelot = "", ""
    energy = None

    # 1. Existing tags
    try:
        m = mutagen.File(full_path)
        if m:
            if not genre:
                g = m.get('genre') or m.get('GENRE') or m.get('TCON') or m.get('\xa9gen')
                if g:
                    genre = str(g[0] if isinstance(g, list) else g)
            k_tag = m.get('TKEY') or m.get('initialkey') or m.get('INITIALKEY') or m.get('key')
            if k_tag:
                key_name, camelot = parse_key_tag(str(k_tag[0] if isinstance(k_tag, list) else k_tag))
            if bpm <= 0:
                b_tag = m.get('TBPM') or m.get('bpm') or m.get('BPM') or m.get('tmpo')
                if b_tag:
                    try:
                        bpm = float(str(b_tag[0] if isinstance(b_tag, list) else b_tag).split()[0])
                    except Exception:
                        pass
    except Exception:
        pass

    # 2. Always decode audio: energy must be measured the same way for every track
    status = 'ok'
    try:
        samples = _decode_slice(full_path)
        if samples is None:
            status = 'error'
        else:
            energy = float(np.sqrt(np.mean(samples ** 2)))
            if bpm <= 0:
                bpm = extract_bpm_aubio(samples, sr=22050)
            if not camelot:
                key_name, camelot = extract_key_from_pcm(samples, sr=22050)
    except Exception:
        status = 'error'

    return (item_id, bpm, key_name, camelot, genre,
            None if energy is None else round(energy, 4), now, status)


def acquire_single_instance_lock():
    """Returns an open lock file, or None if another analyzer is already running."""
    fh = open(LOCK_PATH, 'w')
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def pending_track_ids(navidrome_ids, f_conn) -> set:
    """IDs that still need analysis: never analyzed, or failed more than a day ago."""
    done = set()
    cutoff = time.time() - RETRY_FAILED_AFTER_SECONDS
    for item_id, status, updated_at in f_conn.execute(
            "SELECT item_id, COALESCE(status, 'ok'), COALESCE(updated_at, 0) FROM track_features"):
        if status == 'ok' or updated_at > cutoff:
            done.add(item_id)
    return set(navidrome_ids) - done


def run_analysis(workers: int = 4, limit: int = None):
    lock = acquire_single_instance_lock()
    if lock is None:
        print("[Analyzer] Another analyzer instance is already running. Exiting.")
        return

    try:
        init_features_db()

        # Same DB selection as the engine (primary first), so new uploads get analyzed.
        conn = config.connect_ro(config.resolve_db_path())
        all_tracks = conn.execute("SELECT id, path, bpm, genre FROM media_file").fetchall()
        conn.close()

        f_conn = sqlite3.connect(FEATURES_DB_PATH, timeout=30)
        fc = f_conn.cursor()
        pending = pending_track_ids([t[0] for t in all_tracks], f_conn)
        to_process = [t for t in all_tracks if t[0] in pending]
        if limit:
            to_process = to_process[:limit]

        total = len(all_tracks)
        print(f"[Analyzer] Total library: {total} tracks.")
        print(f"[Analyzer] Already analyzed: {total - len(pending)}. Remaining to process: {len(to_process)}.")

        if not to_process:
            print("[Analyzer] All tracks already analyzed!")
            f_conn.close()
            return

        insert_sql = """
            INSERT OR REPLACE INTO track_features
            (item_id, bpm, key_name, camelot, genre, energy, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        buffer = []
        t_start = time.time()
        processed = 0
        failed = 0

        with mp.Pool(processes=workers) as pool:
            for res in pool.imap_unordered(analyze_track, to_process, chunksize=10):
                buffer.append(res)
                processed += 1
                if res[-1] != 'ok':
                    failed += 1
                if len(buffer) >= 50:
                    fc.executemany(insert_sql, buffer)
                    f_conn.commit()
                    buffer.clear()
                    elapsed = time.time() - t_start
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"[Analyzer] Progress: {processed}/{len(to_process)} "
                          f"({failed} failed) — {rate:.1f} tracks/sec", flush=True)

        if buffer:
            fc.executemany(insert_sql, buffer)
            f_conn.commit()

        f_conn.close()
        print(f"[Analyzer] Finished analyzing {processed} tracks ({failed} missing/failed) "
              f"in {time.time() - t_start:.1f}s.")
    finally:
        lock.close()


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, mp.cpu_count() - 1)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_analysis(workers=workers, limit=limit)
