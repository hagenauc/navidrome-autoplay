import os
import sqlite3
import tempfile
import pytest
from engine import (
    get_primary_artist,
    get_macro_cluster,
    artist_name_token_similarity,
    artist_tokens,
    parse_camelot,
    AutoplayEngine,
)
from analyzer import parse_key_tag


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
