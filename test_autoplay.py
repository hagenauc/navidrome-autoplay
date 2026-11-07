import os
import time
import wave
import sqlite3
import pytest
import numpy as np
from collections import OrderedDict
import config
from engine import (
    get_primary_artist,
    get_macro_cluster,
    artist_name_token_similarity,
    artist_tokens,
    parse_camelot,
    AutoplayEngine,
    build_state,
    _apply_features,
)
from analyzer import (
    parse_key_tag,
    extract_key_from_pcm,
    analyze_track,
    pending_track_ids,
    acquire_single_instance_lock,
)


def test_get_primary_artist():
    cases = [
        ("Daft Punk", "daft punk"),
        ("Taylor Swift", "taylor swift"),
        ("Within Temptation", "within temptation"),
        ("Bill Withers", "bill withers"),
        ("AC/DC", "ac/dc"),
        ("Simon & Garfunkel", "simon & garfunkel"),
        ("Drake feat. Rihanna", "drake"),
        ("Calvin Harris ft. Dua Lipa", "calvin harris"),
        ("David Guetta with Sia", "david guetta"),
        ("Armin van Buuren vs. Sophie Ellis-Bextor", "armin van buuren"),
        ("Queen, David Bowie", "queen"),
    ]
    for raw, expected in cases:
        assert get_primary_artist(raw) == expected, f"Failed on {raw}"


def test_get_macro_cluster():
    # Rock / Pop artists with misleading words in their names must NOT be classical
    assert get_macro_cluster({"artist": "Electric Light Orchestra", "genre": "rock"}) == "rock"
    assert get_macro_cluster({"artist": "Bachman-Turner Overdrive", "genre": "classic rock"}) == "rock"
    assert get_macro_cluster({"artist": "Symphony X", "genre": "metal"}) == "rock"
    assert get_macro_cluster({"artist": "Orchestral Manoeuvres in the Dark", "genre": "electronic"}) == "electronic"
    assert get_macro_cluster({"artist": "Burt Bacharach", "genre": "pop"}) == "pop"

    # Punk / Rock bands must NOT be volksmusik
    assert get_macro_cluster({"artist": "Die Toten Hosen", "genre": "punk rock"}) == "rock"
    assert get_macro_cluster({"artist": "Die Ärzte", "genre": "punk"}) == "rock"

    # Classical markers
    assert get_macro_cluster({"artist": "Berliner Philharmoniker", "genre": ""}) == "classical"
    assert get_macro_cluster({"artist": "Wolfgang Amadeus Mozart", "genre": "classical"}) == "classical"
    assert get_macro_cluster({"artist": "Herbert von Karajan", "genre": ""}) == "classical"

    # Austro-German / Volksmusik
    assert get_macro_cluster({"artist": "Die Zillertaler", "genre": ""}) == "volksmusik_schlager"
    assert get_macro_cluster({"artist": "Andreas Gabalier", "genre": ""}) == "volksmusik_schlager"
    assert get_macro_cluster({"artist": "STS", "genre": "austropop"}) == "volksmusik_schlager"


def test_artist_token_similarity_stops_generic_matches():
    # Common words must NOT create similarity
    assert artist_name_token_similarity("Black Sabbath", "Black Eyed Peas") == 0.0
    assert artist_name_token_similarity("The Beach Boys", "Backstreet Boys") == 0.0
    assert artist_name_token_similarity("Michael Jackson", "George Michael") == 0.0
    assert artist_name_token_similarity("Electric Light Orchestra", "Mahavishnu Orchestra") == 0.0

    # Distinctive regional/brand names should match
    assert artist_name_token_similarity("Die Zillertaler", "Original Zillertaler") == 1.0
    assert artist_name_token_similarity("Seiler und Speer", "Christopher Speer") == 1.0


def test_parse_camelot_and_distance():
    assert parse_camelot("8A") == (8, 0)
    assert parse_camelot("8B") == (8, 1)
    assert parse_camelot("12B") == (12, 1)
    assert parse_camelot("invalid") == (0, 0)

    assert AutoplayEngine._camelot_distance("8A", "8B") == 0.5
    assert AutoplayEngine._camelot_distance("8A", "9A") == 1.0
    assert AutoplayEngine._camelot_distance("8A", "8A") == 0.0
    assert AutoplayEngine._camelot_distance("1A", "12A") == 1.0
    assert AutoplayEngine._camelot_distance("8A", "2A") == 6.0


def test_parse_key_tag():
    assert parse_key_tag("Am") == ("A minor", "8A")
    assert parse_key_tag("C major") == ("C major", "8B")
    assert parse_key_tag("F#m") == ("F# minor", "11A")
    assert parse_key_tag("8A") == ("A minor", "8A")
    assert parse_key_tag("12B") == ("E major", "12B")
    assert parse_key_tag("unknown") == ("", "")


def test_autoplay_engine_full_flow(mock_dbs):
    nav_db, feat_db = mock_dbs
    engine = AutoplayEngine(db_path=nav_db, features_db_path=feat_db)

    assert len(engine.tracks) == 5
    assert len(engine.artist_tracks) == 3

    # Similar tracks for t1 (rock) should recommend rock tracks (t2, t3, t4) and not classical (t5)
    recs = engine.get_similar_tracks("t1", n=3, eliminate_duplicates=True)
    assert len(recs) > 0
    rec_ids = [r["item_id"] for r in recs]
    assert "t1" not in rec_ids
    assert "t5" not in rec_ids  # Guardrail prevents classical mixing with rock

    # Test path finding
    path_res = engine.find_path("t1", "t4", max_steps=5)
    assert "path" in path_res
    assert len(path_res["path"]) >= 2
    assert path_res["path"][0]["item_id"] == "t1"
    assert path_res["path"][-1]["item_id"] == "t4"

    # Test reload
    assert engine.load_database() is True
    assert len(engine.tracks) == 5


# ==============================================================================
# Expanded Tests for TODOS.md Hardening
# ==============================================================================

def test_analyzer_extract_key_from_pcm():
    sr = 22050
    duration = 6.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    # Synthetic A major triad (A4=440Hz, C#5=554.37Hz, E5=659.25Hz)
    chord = (
        np.sin(2 * np.pi * 440.0 * t) +
        np.sin(2 * np.pi * 554.37 * t) +
        np.sin(2 * np.pi * 659.25 * t)
    ).astype(np.float32)

    key_name, camelot = extract_key_from_pcm(chord, sr=sr)
    assert key_name == "A major"
    assert camelot == "11B"

    # Short audio (< 5s) must return empty key
    short_samples = chord[:sr * 4]
    assert extract_key_from_pcm(short_samples, sr=sr) == ("", "")


def test_analyzer_analyze_track_and_path_traversal(tmp_path, monkeypatch):
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    monkeypatch.setattr("analyzer.MUSIC_DIR", str(music_dir))

    # 1. Missing file
    res = analyze_track(("item_missing", "nonexistent.mp3", 120.0, "rock"))
    assert res[-1] == "missing"

    # 2. Path traversal escaping MUSIC_DIR
    outside = tmp_path / "outside.mp3"
    outside.write_text("secret")
    res = analyze_track(("item_traverse", "../outside.mp3", 120.0, "rock"))
    assert res[-1] == "missing"

    res_abs = analyze_track(("item_traverse2", "/etc/passwd", 120.0, "rock"))
    assert res_abs[-1] == "missing"

    # 3. Corrupt audio file (non-decodable bytes)
    corrupt = music_dir / "corrupt.mp3"
    corrupt.write_bytes(b"garbage invalid audio data" * 100)
    res = analyze_track(("item_corrupt", "corrupt.mp3", 120.0, "rock"))
    assert res[-1] == "error"

    # 4. Valid audio file (synthesized wav)
    valid_wav = music_dir / "valid.wav"
    sr = 22050
    t = np.linspace(0, 6, sr * 6, endpoint=False)
    pcm_data = (np.sin(2 * np.pi * 440.0 * t) * 16000).astype(np.int16)

    with wave.open(str(valid_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_data.tobytes())

    res = analyze_track(("item_valid", "valid.wav", 0.0, "jazz"))
    assert res[-1] == "ok"
    assert res[0] == "item_valid"
    assert res[5] is not None and res[5] > 0.0  # energy


def test_analyzer_pending_track_ids(tmp_path):
    feat_db = tmp_path / "features.db"
    conn = sqlite3.connect(str(feat_db))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE track_features (
            item_id TEXT PRIMARY KEY, bpm REAL, key_name TEXT, camelot TEXT,
            genre TEXT, energy REAL, updated_at REAL, status TEXT
        )
    """)
    now = time.time()
    # t1: analyzed ok
    c.execute("INSERT INTO track_features VALUES ('t1', 120, 'Am', '8A', 'rock', 0.5, ?, 'ok')", (now,))
    # t2: failed recently (2 hours ago, within 24h backoff)
    c.execute("INSERT INTO track_features VALUES ('t2', 0, '', '', '', 0, ?, 'error')", (now - 7200,))
    # t3: failed long ago (30 hours ago, exceeds 24h backoff)
    c.execute("INSERT INTO track_features VALUES ('t3', 0, '', '', '', 0, ?, 'error')", (now - 30 * 3600,))
    conn.commit()

    pending = pending_track_ids(["t1", "t2", "t3", "t4_never"], conn)
    conn.close()

    assert "t1" not in pending
    assert "t2" not in pending
    assert "t3" in pending        # retried after 24 hours
    assert "t4_never" in pending  # unanalyzed track


def test_analyzer_acquire_single_instance_lock(tmp_path, monkeypatch):
    lock_file = tmp_path / "test.lock"
    monkeypatch.setattr("analyzer.LOCK_PATH", str(lock_file))

    lock1 = acquire_single_instance_lock()
    assert lock1 is not None

    # Second concurrent attempt must fail
    lock2 = acquire_single_instance_lock()
    assert lock2 is None

    # Releasing the first lock allows acquisition again
    lock1.close()
    lock3 = acquire_single_instance_lock()
    assert lock3 is not None
    lock3.close()


def test_engine_reload_features_incremental(mock_dbs):
    nav_db, feat_db = mock_dbs
    engine = AutoplayEngine(db_path=nav_db, features_db_path=feat_db)

    # Initial BPM for t1 is 120.0
    assert engine.tracks["t1"]["bpm"] == 120.0
    old_mtime = engine.state.features_mtime

    # Update features.db with new BPM and energy for t1
    time.sleep(0.05)
    f_conn = sqlite3.connect(feat_db)
    f_conn.execute("UPDATE track_features SET bpm = 145.0, energy = 0.85 WHERE item_id = 't1'")
    f_conn.commit()
    f_conn.close()

    # Touch features.db to ensure mtime changes
    os.utime(feat_db, (time.time() + 10, time.time() + 10))

    count = engine.reload_features()
    assert count > 0
    assert engine.tracks["t1"]["bpm"] == 145.0
    assert engine.state.features_mtime > old_mtime
    # Embeddings must have been rebuilt
    assert engine.state.embeddings is not None
    assert engine.state.embeddings.shape[0] == 5


def test_session_ttl_and_lru_cap(mock_dbs, monkeypatch):
    nav_db, feat_db = mock_dbs
    engine = AutoplayEngine(db_path=nav_db, features_db_path=feat_db)

    monkeypatch.setattr("config.MAX_SESSIONS", 3)
    monkeypatch.setattr("config.SESSION_TTL_SECONDS", 10)

    # Create 3 sessions
    engine.get_similar_tracks("t1", session_id="s1")
    engine.get_similar_tracks("t1", session_id="s2")
    engine.get_similar_tracks("t1", session_id="s3")

    with engine._sess_lock:
        assert len(engine.sessions) == 3
        assert list(engine.sessions.keys()) == ["s1", "s2", "s3"]

    # 4th session must evict LRU session 's1'
    engine.get_similar_tracks("t1", session_id="s4")
    with engine._sess_lock:
        assert len(engine.sessions) == 3
        assert "s1" not in engine.sessions
        assert "s4" in engine.sessions

    # Test TTL eviction: artificially age session 's2'
    with engine._sess_lock:
        engine.sessions["s2"].last_activity = time.time() - 20

    engine.get_similar_tracks("t1", session_id="s5")
    with engine._sess_lock:
        assert "s2" not in engine.sessions
        assert "s5" in engine.sessions


def test_session_refine_reset_heuristic(mock_dbs):
    nav_db, feat_db = mock_dbs
    engine = AutoplayEngine(db_path=nav_db, features_db_path=feat_db)

    # Start explicit session with seed track t1
    recs = engine.get_similar_tracks("t1", session_id="mix_user_1")
    with engine._sess_lock:
        sess = engine.sessions["mix_user_1"]
        assert sess.seed_id == "t1"
        assert sess.history == ["t1"]

    # User manually jumps / inserts track t3 (which wasn't played yet)
    engine.get_similar_tracks("t3", session_id="mix_user_1")
    with engine._sess_lock:
        sess = engine.sessions["mix_user_1"]
        # Seed anchor must NOT be wiped!
        assert sess.seed_id == "t1"
        assert "t3" in sess.history


def test_max_artist_repeat_and_deduplication(mock_dbs, monkeypatch):
    nav_db, feat_db = mock_dbs

    # Insert a duplicate track (same title & artist as t3 on a Greatest Hits album)
    conn = sqlite3.connect(nav_db)
    conn.execute(
        "INSERT INTO media_file VALUES ('t3_dup', 'Track Three', 'Artist B', 'Greatest Hits', 'a2', 'alb_gh', 2022, 124.0, 190.0, 'CDs/03_gh.mp3', 1, 1, 'rock')"
    )
    conn.commit()
    conn.close()

    engine = AutoplayEngine(db_path=nav_db, features_db_path=feat_db)

    monkeypatch.setattr("config.MAX_ARTIST_REPEAT_IN_QUEUE", 1)
    recs = engine.get_similar_tracks("t1", n=10, eliminate_duplicates=True, session_id="dedup_true")
    artists = [r["author"] for r in recs]
    # No artist should appear more than once when MAX_ARTIST_REPEAT_IN_QUEUE == 1
    for a in set(artists):
        assert artists.count(a) <= 1

    # Now allow multiple tracks per artist and compare eliminate_duplicates=True vs False
    monkeypatch.setattr("config.MAX_ARTIST_REPEAT_IN_QUEUE", 5)
    recs_dedup = engine.get_similar_tracks("t1", n=10, eliminate_duplicates=True, session_id="dedup_1")
    titles_dedup = [r["title"] for r in recs_dedup]
    assert titles_dedup.count("Track Three") == 1

    recs_no_dedup = engine.get_similar_tracks("t1", n=10, eliminate_duplicates=False, session_id="dedup_0")
    titles_no_dedup = [r["title"] for r in recs_no_dedup]
    assert titles_no_dedup.count("Track Three") == 2


def test_empty_and_single_track_libraries(tmp_path):
    # 1. Empty database (0 tracks)
    empty_nav = tmp_path / "empty_nav.db"
    conn = sqlite3.connect(str(empty_nav))
    conn.execute("""
        CREATE TABLE media_file (
            id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT,
            artist_id TEXT, album_id TEXT, year INT, bpm REAL, duration REAL,
            path TEXT, disc_number INT, track_number INT, genre TEXT
        )
    """)
    conn.commit()
    conn.close()

    empty_feat = tmp_path / "empty_feat.db"
    engine_empty = AutoplayEngine(db_path=str(empty_nav), features_db_path=str(empty_feat))
    assert len(engine_empty.tracks) == 0

    assert engine_empty.get_similar_tracks("unknown") == []
    assert engine_empty.find_path("unknown", "unknown") == {"path": []}

    # 2. Single-track library (1 track)
    one_nav = tmp_path / "one_nav.db"
    conn = sqlite3.connect(str(one_nav))
    conn.execute("""
        CREATE TABLE media_file (
            id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT,
            artist_id TEXT, album_id TEXT, year INT, bpm REAL, duration REAL,
            path TEXT, disc_number INT, track_number INT, genre TEXT
        )
    """)
    conn.execute("INSERT INTO media_file VALUES ('solo', 'Solo Track', 'Solo Artist', 'Album', 'a1', 'alb1', 2022, 120.0, 180.0, 'solo.mp3', 1, 1, 'rock')")
    conn.commit()
    conn.close()

    engine_one = AutoplayEngine(db_path=str(one_nav), features_db_path=str(empty_feat))
    assert len(engine_one.tracks) == 1
    # Fallback embedding dimension must be valid
    assert engine_one.state.embeddings.shape == (1, 1)

    # Similar tracks on 1-track library must return empty without error
    assert engine_one.get_similar_tracks("solo") == []

    # Path on single track to itself
    path = engine_one.find_path("solo", "solo")
    assert len(path["path"]) == 1
    assert path["path"][0]["item_id"] == "solo"


def test_native_genre_loading_and_token_matching(tmp_path):
    nav_db = tmp_path / "genre_nav.db"
    conn = sqlite3.connect(str(nav_db))
    conn.execute("""
        CREATE TABLE media_file (
            id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT,
            artist_id TEXT, album_id TEXT, year INT, bpm REAL, duration REAL,
            path TEXT, disc_number INT, track_number INT, genre TEXT
        )
    """)
    # Insert tracks with distinct and compound genres
    tracks = [
        ("p1", "Pop Hit", "Artist P", "Album P", "ap", "albp", 2020, 120.0, 200.0, "p1.mp3", 1, 1, "Pop"),
        ("jp1", "J-Pop Hit", "Artist JP", "Album JP", "ajp", "albjp", 2020, 120.0, 200.0, "jp1.mp3", 1, 1, "J-Pop"),
        ("sp1", "Synth Hit", "Artist SP", "Album SP", "asp", "albsp", 2020, 120.0, 200.0, "sp1.mp3", 1, 1, "Synthpop"),
        ("pr1", "Pop Rocker", "Artist PR", "Album PR", "apr", "albpr", 2020, 120.0, 200.0, "pr1.mp3", 1, 1, "Pop / Rock"),
        ("r1", "Pure Rock", "Artist R", "Album R", "ar", "albr", 2020, 120.0, 200.0, "r1.mp3", 1, 1, "Rock"),
        ("cr1", "Country Rocker", "Artist CR", "Album CR", "acr", "albcr", 2020, 120.0, 200.0, "cr1.mp3", 1, 1, "Country Rock"),
    ]
    conn.executemany("INSERT INTO media_file VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", tracks)
    conn.commit()
    conn.close()

    engine = AutoplayEngine(db_path=str(nav_db), features_db_path="")
    assert engine.tracks["p1"]["genre"] == "Pop"
    assert engine.tracks["jp1"]["genre"] == "J-Pop"
    assert engine.tracks["pr1"]["genre"] == "Pop / Rock"

    st = engine.state
    # Check token-based genre score matching:
    # Pop track (p1) scoring against library
    scores = engine._score(st, st.id_to_idx["p1"], st.id_to_idx["p1"])
    idx_p1 = st.id_to_idx["p1"]
    idx_jp1 = st.id_to_idx["jp1"]
    idx_sp1 = st.id_to_idx["sp1"]
    idx_pr1 = st.id_to_idx["pr1"]

    # Pop / Rock shares the 'pop' token with Pop Hit -> should get genre match bonus
    # J-Pop and Synthpop do NOT share 'pop' as a standalone delimiter token
    # Let's inspect gmatch logic:
    p1_genre_idx = st.a_genre[idx_p1]
    g0_tokens = st.genre_tokens[p1_genre_idx]
    assert "pop" in g0_tokens

    jp1_genre_tokens = st.genre_tokens[st.a_genre[idx_jp1]]
    assert "pop" not in jp1_genre_tokens  # J-Pop token is 'j-pop'
    assert bool(g0_tokens & jp1_genre_tokens) is False

    sp1_genre_tokens = st.genre_tokens[st.a_genre[idx_sp1]]
    assert bool(g0_tokens & sp1_genre_tokens) is False

    pr1_genre_tokens = st.genre_tokens[st.a_genre[idx_pr1]]
    assert "pop" in pr1_genre_tokens  # Pop / Rock splits into {'pop', 'rock'}
    assert bool(g0_tokens & pr1_genre_tokens) is True


def test_energy_normalization_single_and_05(tmp_path):
    feat_db = tmp_path / "test_energy.db"
    conn = sqlite3.connect(str(feat_db))
    conn.execute("""
        CREATE TABLE track_features (
            item_id TEXT PRIMARY KEY, bpm REAL, key_name TEXT, camelot TEXT,
            genre TEXT, energy REAL, updated_at REAL, status TEXT
        )
    """)
    # 1. Test legitimate RMS energy of 0.5000 is NOT dropped when status == 'ok'
    conn.execute("INSERT INTO track_features VALUES ('t_exact_05', 120.0, 'Am', '8A', 'pop', 0.5000, 1000, 'ok')")
    conn.commit()
    conn.close()

    tracks = {
        't_exact_05': {'bpm': 0.0, 'key_name': '', 'camelot': '', 'genre': '', 'energy': None}
    }
    _apply_features(tracks, str(feat_db))

    # Must retain 0.5000 energy and assign 0.5 library percentile for single track
    assert tracks['t_exact_05']['energy'] == 0.5
