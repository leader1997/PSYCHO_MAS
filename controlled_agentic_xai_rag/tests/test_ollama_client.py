"""Tests for Ollama client cooldown."""

from unittest.mock import MagicMock, patch

import controlled_agentic_xai_rag.llm.ollama_client as oc


@patch("controlled_agentic_xai_rag.llm.ollama_client.time.sleep")
@patch("controlled_agentic_xai_rag.llm.ollama_client.requests.post")
def test_chat_applies_cooldown(mock_post, mock_sleep):
    mock_post.return_value = MagicMock(
        status_code=200,
        **{"json.return_value": {"message": {"content": "hello"}}},
    )
    mock_post.return_value.raise_for_status = MagicMock()
    oc.chat("test", system="sys")
    mock_sleep.assert_called_once()


@patch("controlled_agentic_xai_rag.llm.ollama_client.time.sleep")
@patch("controlled_agentic_xai_rag.llm.ollama_client.requests.post")
def test_sequential_calls(mock_post, mock_sleep):
    mock_post.return_value = MagicMock(
        status_code=200,
        **{"json.return_value": {"message": {"content": "ok"}}},
    )
    mock_post.return_value.raise_for_status = MagicMock()
    oc.chat("a")
    oc.chat("b")
    assert mock_sleep.call_count == 2


@patch("controlled_agentic_xai_rag.llm.ollama_client.time.sleep")
@patch("controlled_agentic_xai_rag.llm.ollama_client.requests.post")
def test_embed_splits_on_400(mock_post, mock_sleep):
    import requests

    calls = {"n": 0}

    def side_effect(url, json=None, timeout=None):
        calls["n"] += 1
        m = MagicMock()
        batch = json.get("input")
        size = 1 if isinstance(batch, str) else len(batch)
        if size > 1:
            m.status_code = 400
            m.text = '{"error":"batch too large"}'
            m.json.return_value = {"error": "batch too large"}
            m.raise_for_status.side_effect = requests.HTTPError(response=m)
            return m
        m.status_code = 200
        m.json.return_value = {"embeddings": [[0.1, 0.2]]}
        m.raise_for_status = MagicMock()
        return m

    mock_post.side_effect = side_effect
    with patch.object(oc, "resolve_embed_model", return_value="nomic-embed-text"):
        with patch.object(oc, "ping", return_value=True):
            vecs = oc.embed(["a", "b", "c", "d"], batch_size=4, model="nomic-embed-text")
    assert len(vecs) == 4
    assert calls["n"] > 1
