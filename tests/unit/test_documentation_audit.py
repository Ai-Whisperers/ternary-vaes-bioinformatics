import os

def test_inventory_exists():
    assert os.path.exists("DOCUMENTATION/reports/inventory.md")

def test_inventory_not_empty():
    with open("DOCUMENTATION/reports/inventory.md", "r") as f:
        content = f.read()
    assert len(content) > 0