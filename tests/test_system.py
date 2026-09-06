from osai.system import _platform_support, doctor


def test_doctor_is_serializable():
    report = doctor().as_dict()
    assert report["python"]
    assert report["os"]
    assert isinstance(report["platform_supported"], bool)
    assert isinstance(report["mlx_platform_supported"], bool)


def test_supported_platform_floors(monkeypatch):
    monkeypatch.setattr("osai.hardware.platform.system", lambda: "Darwin")
    monkeypatch.setattr("osai.hardware.platform.mac_ver", lambda: ("11.7", (), ""))
    assert _platform_support("Darwin")[0] is False
    monkeypatch.setattr("osai.hardware.platform.mac_ver", lambda: ("12.0", (), ""))
    assert _platform_support("Darwin")[0] is True

    monkeypatch.setattr("osai.system.platform.release", lambda: "8")
    assert _platform_support("Windows")[0] is False
    monkeypatch.setattr("osai.system.platform.release", lambda: "10")
    assert _platform_support("Windows")[0] is True
    assert _platform_support("Linux")[0] is True
