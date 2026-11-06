import json
import pytest
from server import app, engine


@pytest.fixture
def client(mock_dbs):
    nav_db, feat_db = mock_dbs
    engine.db_path = nav_db
    engine.features_db_path = feat_db
    engine.load_database()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_api_status(client):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "online"
    assert data["total_tracks"] == 5


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


def test_api_reload(client):
    res = client.post("/api/reload")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "reloaded"
