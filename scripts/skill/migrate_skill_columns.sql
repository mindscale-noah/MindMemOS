-- Apply once to an existing database using the previous Skill column names.
-- SQLite and PostgreSQL both preserve stored values and dependent constraints.
BEGIN;
ALTER TABLE skill_versions RENAME COLUMN runtime_type TO implementation;
ALTER TABLE skill_versions RENAME COLUMN runtime_schema_version TO config_schema_version;
ALTER TABLE skill_versions RENAME COLUMN runtime_metadata TO config;
COMMIT;
