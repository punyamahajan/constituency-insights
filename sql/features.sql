DROP VIEW IF EXISTS election_features;
DROP VIEW IF EXISTS constituency_election_summary;

CREATE VIEW constituency_election_summary AS
WITH ranked_candidates AS (
    SELECT
        er.*,
        ROW_NUMBER() OVER (
            PARTITION BY er.constituency_id, er.election_year
            ORDER BY er.vote_share DESC, er.votes DESC, er.result_id
        ) AS vote_rank
    FROM election_result AS er
), election_level AS (
    SELECT
        constituency_id,
        election_year,
        MAX(CASE WHEN vote_rank = 1 THEN party END) AS winner_party,
        MAX(CASE WHEN vote_rank = 1 THEN vote_share END) AS winner_vote_share,
        MAX(CASE WHEN vote_rank = 1 THEN vote_share END)
            - MAX(CASE WHEN vote_rank = 2 THEN vote_share END) AS margin,
        MAX(turnout) AS turnout
    FROM ranked_candidates
    GROUP BY constituency_id, election_year
)
SELECT * FROM election_level;

CREATE VIEW election_features AS
WITH lagged AS (
    SELECT
        summary.*,
        LAG(winner_party, 1) OVER history AS previous_winner_party,
        LAG(winner_party, 2) OVER history AS winner_party_two_elections_ago,
        LAG(winner_vote_share, 1) OVER history AS previous_vote_share,
        LAG(winner_vote_share, 2) OVER history AS vote_share_two_elections_ago,
        LAG(margin, 1) OVER history AS previous_margin,
        LAG(turnout, 1) OVER history AS previous_turnout,
        LAG(turnout, 2) OVER history AS turnout_two_elections_ago
    FROM constituency_election_summary AS summary
    WINDOW history AS (
        PARTITION BY constituency_id ORDER BY election_year
    )
)
SELECT
    lagged.constituency_id,
    c.state_name,
    c.canonical_name,
    lagged.election_year,
    lagged.winner_party AS target_winner_party,
    lagged.winner_vote_share AS target_winner_vote_share,
    lagged.previous_winner_party,
    lagged.previous_vote_share,
    lagged.previous_margin,
    CASE
        WHEN lagged.previous_vote_share IS NULL
          OR lagged.vote_share_two_elections_ago IS NULL THEN NULL
        ELSE lagged.previous_vote_share - lagged.vote_share_two_elections_ago
    END AS swing,
    CASE
        WHEN lagged.previous_winner_party IS NULL
          OR lagged.winner_party_two_elections_ago IS NULL THEN NULL
        -- Whether the party entering T as the previous winner also won T-2.
        -- Both inputs therefore predate the current election target.
        WHEN lagged.previous_winner_party = lagged.winner_party_two_elections_ago THEN 1
        ELSE 0
    END AS incumbent_party_flag,
    CASE
        WHEN lagged.previous_turnout IS NULL
          OR lagged.turnout_two_elections_ago IS NULL THEN NULL
        ELSE lagged.previous_turnout - lagged.turnout_two_elections_ago
    END AS turnout_change,
    d.census_year AS demographics_source_year,
    d.population,
    d.literacy_rate,
    d.urban_share,
    d.scheduled_caste_share
FROM lagged
JOIN constituency AS c USING (constituency_id)
LEFT JOIN demographics AS d
  ON d.constituency_id = lagged.constituency_id
 AND d.census_year = (
     SELECT MAX(d2.census_year)
     FROM demographics AS d2
     WHERE d2.constituency_id = lagged.constituency_id
       AND d2.census_year <= lagged.election_year
 );
