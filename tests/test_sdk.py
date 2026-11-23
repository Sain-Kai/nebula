import pytest
import nebula
from nebula.sdk import NebulaSession, teardown


def test_sdk_teardown():
    session = NebulaSession()
    rep = session.teardown("Quantum mechanics")

    assert rep.title == "Inference Teardown"
    html = rep._repr_html_()
    assert "✦ NEBULA | Inference Teardown" in html
    assert len(str(rep)) > 0


def test_sdk_dla():
    session = NebulaSession()
    rep = session.dla("Test prompt")
    assert rep.title == "Direct Logit Attribution"
    assert "_repr_html_" in dir(rep)


def test_convenience_function():
    rep = teardown("Hello Nebula")
    assert rep.title == "Inference Teardown"
