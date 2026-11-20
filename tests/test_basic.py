from fastapi.testclient import TestClient

from app.main import app
from scrape import extract_page

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_served():
    response = client.get("/")
    assert response.status_code == 200
    assert "DarGlobal" in response.text


def test_chat_validation():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_extract_page():
    html = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>MENU</nav>
            <main>
                <h1>DarGlobal Projects</h1>
                <p>Real estate development content for testing.</p>
            </main>
        </body>
    </html>
    """

    result = extract_page(
        html,
        "https://www.darglobal.co.uk/test",
        "DarGlobal",
    )

    assert result is not None
