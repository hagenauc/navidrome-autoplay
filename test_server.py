import json
import sqlite3
import pytest
import config
from engine import AutoplayEngine
from server import app, engine, check_pending_and_prune_features


@pytest.fixture
def client(mock_dbs):
    nav_db, feat_db = mock_dbs
    engine.db_path = nav_db
    engine.features_db_path = feat_db
    engine.load_database()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_api_status_and_health_no_path_leak(client):
    for endpoint in ["/api/status", "/health"]:
        res = client.get(endpoint)
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "online"
        assert data["total_tracks"] == 5
        # Must not leak the filesystem database path
        assert "database_path" not in data


def test_api_similar_tracks(client):
    # Missing item_id -> 400
    res = client.get("/api/similar_tracks")
    assert res.status_code == 400

    # Valid request
    res = client.get("/api/similar_tracks?item_id=t1&n=2")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)
    assert len(data) <= 2
    if data:
        assert "item_id" in data[0]
        assert "distance" in data[0]


def test_api_similar_artists(client):
    res = client.get("/api/similar_artists")
    assert res.status_code == 400

    res = client.get("/api/similar_artists?artist=Artist%20A&count=2")
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)


def test_api_find_path(client):
    res = client.get("/api/find_path?start_song_id=t1&end_song_id=t4")
    assert res.status_code == 200
    data = res.get_json()
    assert "path" in data
    assert len(data["path"]) >= 2


def test_api_reload_success(client):
    res = client.post("/api/reload")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "reloaded"


def test_api_reload_auth_api_key(client, monkeypatch):
    monkeypatch.setattr("config.AUTOPLAY_API_KEY", "supersecret123")

    # Missing API key -> 401
    res = client.post("/api/reload")
    assert res.status_code == 401

    # Wrong API key -> 401
    res = client.post("/api/reload", headers={"X-API-Key": "wrong"})
    assert res.status_code == 401

    # Valid API key in header -> 200
    res = client.post("/api/reload", headers={"X-API-Key": "supersecret123"})
    assert res.status_code == 200
    assert res.get_json()["status"] == "reloaded"

    # Valid API key in query param -> 200
    res = client.post("/api/reload?api_key=supersecret123")
    assert res.status_code == 200


def test_api_reload_loopback_restriction(client, monkeypatch):
    monkeypatch.setattr("config.AUTOPLAY_API_KEY", "")

    # Request from non-loopback IP without API key -> 403 Forbidden
    res = client.post("/api/reload", environ_overrides={"REMOTE_ADDR": "192.168.1.100"})
    assert res.status_code == 403

    # Request from loopback IP -> 200 OK
    res = client.post("/api/reload", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
    assert res.status_code == 200


def test_api_reload_error_handling(client, monkeypatch):
    monkeypatch.setattr("config.AUTOPLAY_API_KEY", "")

    def mock_failing_reload(*args, **kwargs):
        raise RuntimeError("database disk image is malformed / sensitive internal details")

    monkeypatch.setattr(engine, "load_database", mock_failing_reload)

    res = client.post("/api/reload", environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
    assert res.status_code == 500
    data = res.get_json()
    # Must return generic error, without leaking internal details or tracebacks
    assert data == {"error": "Internal server error"}


def test_check_pending_and_prune_features(mock_dbs):
    nav_db, feat_db = mock_dbs

    # Add an orphaned track to track_features and a new unanalyzed track to media_file
    fconn = sqlite3.connect(feat_db)
    fconn.execute(
        "INSERT INTO track_features (item_id, bpm, key_name, camelot, genre, energy, updated_at, status) "
        "VALUES ('orphan_track', 120.0, 'Am', '8A', 'rock', 0.5, 1000.0, 'ok')"
    )
    fconn.commit()
    fconn.close()

    nconn = sqlite3.connect(nav_db)
    nconn.execute(
        "INSERT INTO media_file VALUES ('new_track', 'New Song', 'Artist A', 'Album 1', 'a1', 'al1', 2024, 125.0, 200.0, 'new.mp3', 1, 5, 'rock')"
    )
    nconn.commit()
    nconn.close()

    pending = check_pending_and_prune_features(nav_db, feat_db)
    assert "new_track" in pending
    assert "orphan_track" not in pending

    # Verify orphan_track was deleted from track_features
    fconn = sqlite3.connect(feat_db)
    remaining = {r[0] for r in fconn.execute("SELECT item_id FROM track_features").fetchall()}
    fconn.close()
    assert "orphan_track" not in remaining


def test_api_empty_and_single_track_endpoints(tmp_path, monkeypatch):
    # 0-track database via HTTP API
    empty_nav = tmp_path / "empty_nav.db"
    conn = sqlite3.connect(str(empty_nav))
    conn.execute("""
        CREATE TABLE media_file (
            id TEXT PRIMARY KEY, title TEXT, artist TEXT, album TEXT,
            artist_id TEXT, album_id TEXT, year INT, bpm REAL, duration REAL,
            path TEXT, disc_number INT, track_number INT
        )
    """)
    conn.commit()
    conn.close()

    empty_feat = tmp_path / "empty_feat.db"
    monkeypatch.setattr(engine, "db_path", str(empty_nav))
    monkeypatch.setattr(engine, "features_db_path", str(empty_feat))
    engine.load_database()
    app.config["TESTING"] = True

    with app.test_client() as c:
        res_sim = c.get("/api/similar_tracks?item_id=solo")
        assert res_sim.status_code == 200
        assert res_sim.get_json() == []

        res_path = c.get("/api/find_path?start_song_id=solo&end_song_id=solo")
        assert res_path.status_code == 200
        assert res_path.get_json() == {"path": []}

    # 1-track database via HTTP API
    conn = sqlite3.connect(str(empty_nav))
    conn.execute(
        "INSERT INTO media_file VALUES ('solo', 'Solo Song', 'Solo Artist', 'Solo Album', 'sa1', 'sal1', 2022, 120.0, 180.0, 'solo.mp3', 1, 1)"
    )
    conn.commit()
    conn.close()
    engine.load_database()

    with app.test_client() as c:
        res_sim = c.get("/api/similar_tracks?item_id=solo")
        assert res_sim.status_code == 200
        assert res_sim.get_json() == []

        res_path = c.get("/api/find_path?start_song_id=solo&end_song_id=solo")
        assert res_path.status_code == 200
        assert len(res_path.get_json()["path"]) == 1
