PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS constituency (
    constituency_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    state_name TEXT NOT NULL,
    valid_from_year INTEGER,
    valid_to_year INTEGER,
    UNIQUE (state_name, canonical_name, valid_from_year)
);

CREATE TABLE IF NOT EXISTS constituency_name_mapping (
    mapping_id INTEGER PRIMARY KEY,
    constituency_id INTEGER NOT NULL REFERENCES constituency(constituency_id),
    state_name TEXT NOT NULL,
    source_name TEXT NOT NULL,
    valid_from_year INTEGER,
    valid_to_year INTEGER,
    mapping_type TEXT NOT NULL,
    UNIQUE (state_name, source_name, valid_from_year)
);

CREATE TABLE IF NOT EXISTS election_result (
    result_id INTEGER PRIMARY KEY,
    constituency_id INTEGER NOT NULL REFERENCES constituency(constituency_id),
    election_year INTEGER NOT NULL,
    candidate_name TEXT NOT NULL,
    party TEXT NOT NULL,
    votes INTEGER NOT NULL CHECK (votes >= 0),
    vote_share REAL NOT NULL CHECK (vote_share BETWEEN 0 AND 100),
    turnout REAL CHECK (turnout BETWEEN 0 AND 100),
    is_winner INTEGER NOT NULL CHECK (is_winner IN (0, 1)),
    UNIQUE (constituency_id, election_year, candidate_name, party)
);

CREATE TABLE IF NOT EXISTS demographics (
    demographic_id INTEGER PRIMARY KEY,
    constituency_id INTEGER NOT NULL REFERENCES constituency(constituency_id),
    census_year INTEGER NOT NULL,
    population INTEGER CHECK (population IS NULL OR population >= 0),
    literacy_rate REAL CHECK (literacy_rate IS NULL OR literacy_rate BETWEEN 0 AND 100),
    urban_share REAL CHECK (urban_share IS NULL OR urban_share BETWEEN 0 AND 100),
    scheduled_caste_share REAL CHECK (
        scheduled_caste_share IS NULL OR scheduled_caste_share BETWEEN 0 AND 100
    ),
    UNIQUE (constituency_id, census_year)
);

CREATE INDEX IF NOT EXISTS idx_result_constituency_year
    ON election_result (constituency_id, election_year);
CREATE INDEX IF NOT EXISTS idx_demographics_constituency_year
    ON demographics (constituency_id, census_year);
CREATE INDEX IF NOT EXISTS idx_mapping_source
    ON constituency_name_mapping (state_name, source_name);
