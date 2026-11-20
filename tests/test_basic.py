from fastapi.testclient import TestClient

from app.main import app
from scraper import clean

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_index_served():
    assert "DarGlobal" in client.get("/").text


def test_chat_validation():
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_clean_strips_noise():
    html = "<html><title>T</title><nav>MENU</nav><body><p>Real content here</p></body></html>"
    title, text = clean(html)
    assert title == "T" and "Real content here" in text and "MENU" not in text
