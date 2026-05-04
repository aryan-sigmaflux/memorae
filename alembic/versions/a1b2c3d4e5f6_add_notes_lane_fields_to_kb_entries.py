"""Add notes lane fields to kb_entries

Revision ID: a1b2c3d4e5f6
Revises: 45857cec00ca
Create Date: 2026-05-04 16:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '45857cec00ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add context_clues (TEXT[]), status, and source_message to kb_entries."""
    # context_clues as a proper Postgres array, not plain text
    op.execute("ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS context_clues TEXT[] DEFAULT '{}'")
    op.execute("ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active'")
    op.execute("ALTER TABLE kb_entries ADD COLUMN IF NOT EXISTS source_message TEXT")


def downgrade() -> None:
    """Remove notes lane columns."""
    op.execute("ALTER TABLE kb_entries DROP COLUMN IF EXISTS context_clues")
    op.execute("ALTER TABLE kb_entries DROP COLUMN IF EXISTS status")
    op.execute("ALTER TABLE kb_entries DROP COLUMN IF EXISTS source_message")
