"""Add pgvector embedding to kb_entries

Revision ID: 45857cec00ca
Revises: 
Create Date: 2026-03-21 11:20:14.421380

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '45857cec00ca'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    op.execute('ALTER TABLE kb_entries DROP COLUMN IF EXISTS embedding')
    op.execute('ALTER TABLE kb_entries ADD COLUMN embedding vector(768)')


def downgrade() -> None:
    """Downgrade schema."""
    op.execute('ALTER TABLE kb_entries DROP COLUMN IF EXISTS embedding')
    op.execute('ALTER TABLE kb_entries ADD COLUMN embedding JSONB')
    # Be careful with dropping the extension if other tables use it, but since we just added it:
    op.execute('DROP EXTENSION IF EXISTS "vector"')
