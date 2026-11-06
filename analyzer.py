import os
import sys
import time
import sqlite3
import subprocess
import multiprocessing as mp
import numpy as np
import scipy.signal
import aubio
import mutagen
from mutagen.id3 import ID3
import config

MUSIC_DIR = "/opt/navidrome/music"
FEATURES_DB_PATH = "/root/navidrome-autoplay/features.db"

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
    conn = sqlite3.connect(FEATURES_DB_PATH)
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
    conn.commit()
    conn.close()

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

def analyze_track(item):
    item_id, rel_path, db_bpm, db_genre = item
    full_path = os.path.join(MUSIC_DIR, rel_path)

    if not os.path.exists(full_path):
        return None

    genre = db_genre or ""
    bpm = float(db_bpm) if db_bpm else 0.0
    key_name = ""
    camelot = ""
    energy = 0.5

    # 1. Mutagen tag inspection for existing ID3 tags
    try:
        m = mutagen.File(full_path)
        if m:
            if not genre:
                g = m.get('genre') or m.get('GENRE') or m.get('TCON') or m.get('\xa9gen')
                if g:
                    if isinstance(g, list):
                        genre = str(g[0])
                    else:
                        genre = str(g)

            # Check for existing key tag
            k_tag = m.get('TKEY') or m.get('initialkey') or m.get('key')
            if k_tag:
                raw_k = str(k_tag[0] if isinstance(k_tag, list) else k_tag).strip()
                if raw_k in CAMELOT_MAP.values():
                    camelot = raw_k
                elif raw_k in CAMELOT_MAP:
                    key_name = raw_k
                    camelot = CAMELOT_MAP[raw_k]

            # Check for existing BPM tag
            if bpm <= 0:
                b_tag = m.get('TBPM') or m.get('bpm') or m.get('tmpo')
                if b_tag:
                    try:
                        bpm = float(str(b_tag[0] if isinstance(b_tag, list) else b_tag).split()[0])
                    except Exception:
                        pass
    except Exception:
        pass

    # 2. If BPM or Key still missing, decode audio slice via FFmpeg
    if bpm <= 0 or not camelot:
        try:
            # Grab 40s from offset 25s for reliable groove/harmony
            cmd = [
                'ffmpeg', '-ss', '25', '-t', '40', '-i', full_path,
                '-f', 'f32le', '-ac', '1', '-ar', '22050', '-v', 'error', '-'
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw, _ = proc.communicate(timeout=8)
            samples = np.frombuffer(raw, dtype=np.float32)

            if len(samples) > 22050 * 5:
                # RMS Energy
                energy = float(np.sqrt(np.mean(samples ** 2)))

                if bpm <= 0:
                    bpm = extract_bpm_aubio(samples, sr=22050)

                if not camelot:
                    key_name, camelot = extract_key_from_pcm(samples, sr=22050)
        except Exception:
            pass

    return (item_id, bpm, key_name, camelot, genre, round(energy, 4), time.time())

def run_analysis(workers: int = 4, limit: int = None):
    init_features_db()

    conn = sqlite3.connect(config.BACKUP_DB_PATH if os.path.exists(config.BACKUP_DB_PATH) else config.PRIMARY_DB_PATH)
    c = conn.cursor()

    # Get already processed IDs
    f_conn = sqlite3.connect(FEATURES_DB_PATH)
    fc = f_conn.cursor()
    already_done = set(r[0] for r in fc.execute("SELECT item_id FROM track_features").fetchall())

    query = "SELECT id, path, bpm, genre FROM media_file"
    all_tracks = c.execute(query).fetchall()
    conn.close()

    to_process = [t for t in all_tracks if t[0] not in already_done]
    if limit:
        to_process = to_process[:limit]

    total = len(all_tracks)
    done_count = len(already_done)
    pending = len(to_process)

    print(f"[Analyzer] Total library: {total} tracks.")
    print(f"[Analyzer] Already analyzed: {done_count}. Remaining to process: {pending}.")

    if not to_process:
        print("[Analyzer] All tracks already analyzed!")
        f_conn.close()
        return

    batch_size = 50
    buffer = []
    t_start = time.time()
    processed_in_run = 0

    with mp.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(analyze_track, to_process, chunksize=10):
            if res is not None:
                buffer.append(res)
                processed_in_run += 1

                if len(buffer) >= batch_size:
                    fc.executemany("""
                        INSERT OR REPLACE INTO track_features 
                        (item_id, bpm, key_name, camelot, genre, energy, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, buffer)
                    f_conn.commit()
                    buffer.clear()

                    elapsed = time.time() - t_start
                    rate = processed_in_run / elapsed if elapsed > 0 else 0
                    pct = ((done_count + processed_in_run) / total) * 100.0
                    print(f"[Analyzer] Progress: {done_count + processed_in_run}/{total} ({pct:.1f}%) — {rate:.1f} tracks/sec")

        if buffer:
            fc.executemany("""
                INSERT OR REPLACE INTO track_features 
                (item_id, bpm, key_name, camelot, genre, energy, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, buffer)
            f_conn.commit()
            buffer.clear()

    f_conn.close()
    print(f"[Analyzer] Finished analyzing {processed_in_run} tracks in {time.time() - t_start:.1f}s.")

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, mp.cpu_count() - 1)
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    run_analysis(workers=workers, limit=limit)
