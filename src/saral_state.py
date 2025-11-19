# saral_state.py
ACTIVE_COLLECTION = None

def set_active_collection(name: str):
    """Sets the active paper collection name."""
    global ACTIVE_COLLECTION
    ACTIVE_COLLECTION = name
    print(f"📘 Active collection set to: {ACTIVE_COLLECTION}")

def get_active_collection():
    """Returns the current active paper collection."""
    return ACTIVE_COLLECTION
