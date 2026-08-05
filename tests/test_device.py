from tristatelite.device import select_device


def test_select_device_returns_supported_device():
    assert select_device().type in {"mps", "cuda", "cpu"}
