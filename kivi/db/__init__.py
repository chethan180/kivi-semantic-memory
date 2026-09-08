"""Storage layer: connection handling, migrations, and schema."""

from kivi.db.connection import connect, vec_available
from kivi.db.migrate import current_version, migrate, pending_migrations

__all__ = [
    "connect",
    "vec_available",
    "migrate",
    "current_version",
    "pending_migrations",
]
