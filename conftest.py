import sqlite3
import pytest


@pytest.fixture
def mock_dbs(tmp_path):
    nav_db = tmp_path / "navidrome.db"
    feat_db = tmp_path / "features.db"

    conn = sqlite3.connect(str(nav_db))
    c = conn.cursor()
    c.execute("""
        CREATE TABLE media_file (
            id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT,
            artist_id TEXT, album_id TEXT, year INT, bpm REAL, duration REAL,
            path TEXT, disc_number INT, track_number INT, genre TEXT
        )
    """)
    c.execute("""
        CREATE TABLE annotation (
            item_id TEXT, item_type TEXT, play_count INT, starred INT, user_id TEXT
        )
    """)
    c.execute("""
        CREATE TABLE scrobbles (
            user_id TEXT, media_file_id TEXT, submission_time INT
        )
    """)
    c.execute("""
        CREATE TABLE playlist_tracks (
            id INT PRIMARY KEY, playlist_id TEXT, media_file_id TEXT
        )
    """)
    c.execute("""
        CREATE TABLE artist (
            id TEXT PRIMARY KEY, name TEXT, similar_artists TEXT
        )
    """)

    # Populate 5 tracks across 2 artists and 2 albums
    tracks = [
        ("t1", "Track One", "Artist A", "Album 1", "a1", "alb1", 2020, 120.0, 200.0, "CDs/01.mp3", 1, 1, "rock"),
        ("t2", "Track Two", "Artist A", "Album 1", "a1", "alb1", 2020, 122.0, 210.0, "CDs/02.mp3", 1, 2, "rock"),
        ("t3", "Track Three", "Artist B", "Album 2", "a2", "alb2", 2021, 124.0, 190.0, "CDs/03.mp3", 1, 1, "rock"),
        ("t4", "Track Four", "Artist B", "Album 2", "a2", "alb2", 2021, 128.0, 220.0, "CDs/04.mp3", 1, 2, "rock"),
        ("t5", "Classical Piece", "Composer C", "Album 3", "a3", "alb3", 1980, 80.0, 400.0, "CDs/05.mp3", 1, 1, "classical"),
    ]
    for t in tracks:
        c.execute("INSERT INTO media_file VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", t)

    # Scrobbles: session with t1, t2, t3
    c.execute("INSERT INTO scrobbles VALUES ('u1', 't1', 1000)")
    c.execute("INSERT INTO scrobbles VALUES ('u1', 't2', 1200)")
    c.execute("INSERT INTO scrobbles VALUES ('u1', 't3', 1400)")

    conn.commit()
    conn.close()

    # Features db
    f_conn = sqlite3.connect(str(feat_db))
    fc = f_conn.cursor()
    fc.execute("""
        CREATE TABLE track_features (
            item_id TEXT PRIMARY KEY, bpm REAL, key_name TEXT, camelot TEXT,
            genre TEXT, energy REAL, updated_at REAL, status TEXT
        )
    """)
    fc.execute("INSERT INTO track_features VALUES ('t1', 120.0, 'A minor', '8A', 'rock', 0.25, 1000, 'ok')")
    fc.execute("INSERT INTO track_features VALUES ('t2', 122.0, 'C major', '8B', 'rock', 0.26, 1000, 'ok')")
    fc.execute("INSERT INTO track_features VALUES ('t3', 124.0, 'A minor', '8A', 'rock', 0.24, 1000, 'ok')")
    fc.execute("INSERT INTO track_features VALUES ('t4', 128.0, 'E minor', '9A', 'rock', 0.30, 1000, 'ok')")
    fc.execute("INSERT INTO track_features VALUES ('t5', 80.0, 'D minor', '7A', 'classical', 0.10, 1000, 'ok')")
    f_conn.commit()
    f_conn.close()

    return str(nav_db), str(feat_db)
