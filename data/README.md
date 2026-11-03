# Election and demographic data

This repository does not download, redistribute, or invent election or census
records. Obtain each source under its own terms, preserve the original download
outside version control, and prepare the local CSVs described below.

## Lok Sabha election results

Obtain Lok Sabha constituency-level candidate results from the Trivedi Centre
for Political Data (TCPD) Lok Dhaba data portal:

- Project page: <https://tcpd.ashoka.edu.in/lok-dhaba/>
- TCPD data access portal: <https://tcpd.ashoka.edu.in/>

Review TCPD's current access, citation, and redistribution conditions before
downloading. Place the locally prepared file at:

`data/raw/elections/lok_dhaba_results.csv`

Required columns:

| Column | Meaning |
| --- | --- |
| `state_name` | State or union territory name used by the source |
| `constituency_name` | Source constituency name |
| `election_year` | Four-digit Lok Sabha election year |
| `candidate_name` | Candidate name |
| `party` | Party name or code |
| `votes` | Candidate votes |
| `vote_share` | Candidate vote share on a 0–100 scale |
| `turnout` | Constituency turnout on a 0–100 scale |
| `is_winner` | `0` or `1`; exactly one winner per constituency-year |

## Census and demographic data

Obtain constituency-compatible demographic variables from an authoritative
source such as the Census of India or a documented constituency-to-census
crosswalk. Start from the Census of India data catalogue:

- <https://censusindia.gov.in/census.website/data/census-tables>

Lok Sabha constituencies are electoral geographies and do not generally align
directly with census administrative units. Record and review the aggregation or
crosswalk methodology; do not treat name similarity as geographic equivalence.
Place the prepared file at:

`data/raw/demographics/constituency_demographics.csv`

Required columns are `state_name`, `canonical_name`, `census_year`,
`population`, `literacy_rate`, `urban_share`, and `scheduled_caste_share`.
Percentage fields use a 0–100 scale. Additional source columns are ignored.

## Reviewed constituency name and delimitation mappings

Create the manually reviewed mapping file:

`data/raw/elections/constituency_name_mapping.csv`

Required columns are `state_name`, `source_name`, `canonical_name`,
`valid_from_year`, `valid_to_year`, and `mapping_type`. Use `mapping_type` to
record values such as `same`, `renamed`, `redrawn`, or `delimitation`. Empty
validity bounds mean open-ended. A mapping asserts an identity decision and
must be reviewed by a person familiar with the relevant delimitation history.

The build script never accepts a fuzzy match automatically. For any unmatched
result name it writes a suggested candidate and score to the Git-ignored file
`outputs/local/unmatched_constituencies.csv`, skips that result row, and requires
the mapping CSV to be updated before a complete build.

## Build

After placing all three files locally:

```powershell
.\.venv\Scripts\python.exe scripts\build_election_db.py
```

The default database is `outputs/local/constituency_elections.sqlite`. Missing
inputs cause an actionable error; no network request or automatic download is
performed.
