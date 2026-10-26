from agents.tools import check_shipment_status


def test_known_tracking_id_returns_status():
    result = check_shipment_status.invoke({"tracking_ref": "TRK1001"})
    assert result["status"] == "in_transit"


def test_lowercase_and_whitespace_normalize_the_same():
    exact = check_shipment_status.invoke({"tracking_ref": "TRK1001"})
    messy = check_shipment_status.invoke({"tracking_ref": "  trk1001  "})
    assert exact == messy


def test_unknown_tracking_id_returns_not_found():
    result = check_shipment_status.invoke({"tracking_ref": "NOT-A-REAL-ID"})
    assert result["status"] == "not_found"
