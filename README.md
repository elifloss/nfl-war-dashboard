NFL WAR Dashboard 🏈
A position-neutral, team-adjusted Wins Above Replacement (WAR) model for NFL skill position players — built entirely on public play-by-play data.
Live dashboard coming soon

What This Is
Most NFL analytics tools evaluate players in isolation. This model accounts for the fact that a receiver's production is partially a function of his quarterback, and a running back's yards are partly a function of his offensive line.
The WAR model isolates individual contribution by:

Computing position-specific performance scores from EPA (Expected Points Added) and CPOE (Completion Percentage Over Expected)
Adjusting each player's score against their team's average at that position group, separating individual talent from supporting cast
Setting replacement level at the 15th percentile of qualified players per position
Scaling final WAR values against historical team win totals to ensure the numbers are meaningful in real football terms

The result is a single comparable number across QB, RB, WR, and TE — answering the question: how many wins does this player add over a freely available replacement?

Features

Leaderboard — Top players by WAR, filterable by position, team, and WAR range
Team View — Total WAR by franchise, showing which organizations accumulate the most talent value
Player Explorer — Search any player for their WAR, EPA/play, and position-adjusted score
Methodology Tab — Full model documentation so the numbers are reproducible and auditable


Model Details
PositionPrimary MetricMinimum ThresholdQBEPA/dropback + CPOE100 dropbacks/seasonRBEPA/carry50 carries/seasonWR/TEEPA/target + YAC30 targets/season
Position weights: QB (0.22) · WR/TE (0.14) · RB (0.10)
Team adjustment: 70% residual from position-group regression + 30% raw score
Data: nfl_data_py · Seasons 2022–2024 · Fully public, no proprietary grades required

Run Locally
bashpip install streamlit nfl-data-py pandas numpy scikit-learn plotly
streamlit run app.py

Stack
Python · Streamlit · nfl_data_py · pandas · scikit-learn · Plotly

Background
This dashboard extends research originally conducted as part of a collaborative NFL analytics project at Penn State University, where the core WAR methodology was developed using PFF grade data and multilevel regression. This implementation reproduces and expands that methodology using fully public play-by-play data.

Author
Eli Floss · Penn State Data Science · NFL WAR Dashboard 🏈
A position-neutral, team-adjusted Wins Above Replacement (WAR) model for NFL skill position players — built entirely on public play-by-play data.
Live dashboard coming soon

What This Is
Most NFL analytics tools evaluate players in isolation. This model accounts for the fact that a receiver's production is partially a function of his quarterback, and a running back's yards are partly a function of his offensive line.
The WAR model isolates individual contribution by:

Computing position-specific performance scores from EPA (Expected Points Added) and CPOE (Completion Percentage Over Expected)
Adjusting each player's score against their team's average at that position group, separating individual talent from supporting cast
Setting replacement level at the 15th percentile of qualified players per position
Scaling final WAR values against historical team win totals to ensure the numbers are meaningful in real football terms

The result is a single comparable number across QB, RB, WR, and TE — answering the question: how many wins does this player add over a freely available replacement?

Features

Leaderboard — Top players by WAR, filterable by position, team, and WAR range
Team View — Total WAR by franchise, showing which organizations accumulate the most talent value
Player Explorer — Search any player for their WAR, EPA/play, and position-adjusted score
Methodology Tab — Full model documentation so the numbers are reproducible and auditable


Model Details
PositionPrimary MetricMinimum ThresholdQBEPA/dropback + CPOE100 dropbacks/seasonRBEPA/carry50 carries/seasonWR/TEEPA/target + YAC30 targets/season
Position weights: QB (0.22) · WR/TE (0.14) · RB (0.10)
Team adjustment: 70% residual from position-group regression + 30% raw score
Data: nfl_data_py · Seasons 2022–2024 · Fully public, no proprietary grades required

Run Locally
bashpip install streamlit nfl-data-py pandas numpy scikit-learn plotly
streamlit run app.py

Stack
Python · Streamlit · nfl_data_py · pandas · scikit-learn · Plotly

Background
This dashboard extends research originally conducted as part of a collaborative NFL analytics project at Penn State University, where the core WAR methodology was developed using PFF grade data and multilevel regression. This implementation reproduces and expands that methodology using fully public play-by-play data.

Author
Elijah Legall · Penn State Data Science · https://www.linkedin.com/in/elijah-legall-8aa53b261/
