-- Prepare an existing local skill_versions table before ensure_collection().
-- Safe to repeat on SQLite and PostgreSQL; no version rows are modified.
BEGIN;
CREATE UNIQUE INDEX IF NOT EXISTS skill_versions_owner_version_uq
    ON skill_versions (project_id, skill_id, version_id);
COMMIT;
