"""Tests for the deterministic listing check in fact_check.py.

These run without a model or an API key:

    cd resale_listing_graph/tests
    python -m pytest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fact_check import check_listing  # noqa: E402

SELLER_FACTS = """
item_type: dangle earrings
color: red and white
size_or_dimensions: 1.5 inches
material: glass beads
condition: preowned, excellent condition
flaws: None
seller_notes: handmade, heart design
"""

GOOD_DRAFT = {
    "title": "Red and White Glass Bead Heart Dangle Earrings",
    "category": "Jewelry",
    "condition": "Preowned, excellent condition",
    "description": "Handmade heart design made of glass beads, 1.5 inches long. No flaws.",
}


def test_draft_that_sticks_to_the_facts_passes():
    assert check_listing(GOOD_DRAFT, SELLER_FACTS, required_text=SELLER_FACTS) == []


def test_price_in_text_is_rejected():
    draft = dict(GOOD_DRAFT, description=GOOD_DRAFT["description"] + " Only $15.")
    assert any("price" in issue for issue in check_listing(draft, SELLER_FACTS))


def test_unsupported_claim_and_filler_words_are_rejected():
    draft = dict(GOOD_DRAFT, title="Vintage Charming Glass Bead Earrings")
    issues = check_listing(draft, SELLER_FACTS)
    assert any('"vintage"' in issue for issue in issues)
    assert any('"charming"' in issue for issue in issues)


def test_number_not_in_facts_is_rejected():
    draft = dict(GOOD_DRAFT, description="Handmade heart design made of glass beads, 2 inches long. No flaws.")
    assert any("number 2 " in issue for issue in check_listing(draft, SELLER_FACTS))


def test_mentioning_the_seller_is_rejected():
    draft = dict(GOOD_DRAFT, description="Handmade heart design made of glass beads, 1.5 inches long. No flaws noted by the seller.")
    assert any("Do not mention the seller" in issue for issue in check_listing(draft, SELLER_FACTS))


def test_missing_selling_point_is_rejected():
    draft = dict(GOOD_DRAFT, description="Heart design made of glass beads, 1.5 inches long. No flaws.")
    issues = check_listing(draft, SELLER_FACTS, required_text=SELLER_FACTS)
    assert any('"handmade"' in issue for issue in issues)


def test_negated_selling_point_is_not_required():
    facts = SELLER_FACTS.replace("handmade, heart design", "not handmade, heart design")
    draft = dict(GOOD_DRAFT, description="Heart design made of glass beads, 1.5 inches long. No flaws.")
    assert check_listing(draft, facts, required_text=facts) == []
