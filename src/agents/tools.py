"""The tracking tool: mocked shipment data.

Same honest simplification as Project 1's account-status tool (docs/ROADMAP.md
Sec. 7 item 1) — deterministic fake data, not a real carrier/USPS API. Four
scenarios on purpose, covering the kinds of complaints in Sec. 2 of the roadmap.
"""

from __future__ import annotations

from langchain_core.tools import tool
# @tool turns the plain function into a LangChain tool. 
# The name, the argument type, and the docstring are what get sent to the LLM.
_MOCK_SHIPMENTS: dict[str, dict] = {
    "TRK1001": {
        "status": "in_transit",
        "location": "Memphis, TN distribution center",
        "last_update": "2026-09-30",
        "note": "On schedule, no delays reported.",
    },
    "TRK1002": {
        "status": "customs_hold",
        "location": "JFK International Mail Facility",
        "last_update": "2026-09-27",
        "note": (
            "Held for customs inspection. Declared value is near the $2,500 "
            "formal-entry threshold; a formal entry may be required before release."
        ),
    },
    "TRK1003": {
        "status": "delivered",
        "location": "Front door (per carrier GPS)",
        "last_update": "2026-09-20",
        "note": "Marked delivered by the carrier. No recipient signature on file.",
    },
    "TRK1004": {
        "status": "failed_delivery_attempt",
        "location": "Local depot",
        "last_update": "2026-10-01",
        "note": "Delivery attempted, no one available to receive it. Will retry next business day.",
    },
}


@tool
def check_shipment_status(tracking_ref: str) -> dict:
    """Look up the current status of a shipment by its tracking number.

    Returns the status, last known location, last update date, and a note. If the
    tracking number isn't recognized, returns a not-found result instead of guessing.
    """
    shipment = _MOCK_SHIPMENTS.get(tracking_ref.strip().upper())
    if shipment is None:
        return {"status": "not_found", "note": f"No shipment found for tracking number {tracking_ref!r}."}
    return shipment
