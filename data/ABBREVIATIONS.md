# AFL Data Abbreviations and Naming Conventions

This document provides a comprehensive reference for all abbreviations and naming conventions used in the AFL prediction system data files.

## General

| Abbreviation | Description |
|--------------|-------------|
| Age | Age as at 31 December |
| Player Rating | Official AFL Player Ratings (Overview) |
| CV Avg | Average Coaches Votes per match |
| TOG | Time on ground percentage |

## Disposals

| Abbreviation | Description |
|--------------|-------------|
| Kick | Kicks |
| HB | Handballs |
| Dis. | Disposals |
| Eff % | Disposal Efficiency (minimum 20 disposals) |
| Kick Eff % | Kick Efficiency (minimum 20 kicks) |
| HB Eff % | Handball Efficiency (minimum 20 handballs) |
| Kick % | Kicks / Disposals (minimum 20 disposals) |
| In 50s | Inside 50s |
| Reb 50s | Rebound 50s |
| Mtrs Gnd | Metres gained |
| Mtrs / D | Metres gained / Disposal (minimum 20 disposals) |
| Clng | Clangers |
| D/Cg | Disposals per clanger |
| TO | Turnovers |
| D/TO | Disposals per turnover |

## Possessions

| Abbreviation | Description |
|--------------|-------------|
| CP | Contested possessions |
| UP | Uncontested possessions |
| Poss | Total possessions |
| CP% | Contested possessions / total possessions (minimum 20 total possessions) |
| Int Poss | Intercept possessions |
| GB Gets | Ground ball gets |
| F50 GBG | Ground ball gets in the forward 50 |
| Hard Gets | Hard ball gets. A disputed ball at ground level under direct physical pressure or out of a ruck contest, resulting in an opportunity to effect a legal disposal. |
| Loose Gets | Loose ball gets. A disputed ball at ground level not under direct physical pressure that results in an opportunity to record a legal disposal. |
| Post Cl CP | Post-clearance contested possessions |
| Post Cl GBG | Post-clearance ground ball gets |
| Gath HO | Gathers from hitout. A possession gained from a teammate's hitout-to-advantage |
| Crumb | Crumbing possessions. Ground ball gets that are won by a player at ground level after a marking contest. |
| HB Rec | Handball receives |

## Clearances

| Abbreviation | Description |
|--------------|-------------|
| CBA | Centre bounce attendances |
| CBA % | Centre bounce attendances / total centre bounces |
| Ctr Clr | Centre clearances |
| CC / CBA | Centre clearances / centre bounce attendances (minimum 20 centre bounce attendances) |
| Stp Clr | Stoppage clearances |
| Tot | Total clearances |

## Marks

| Abbreviation | Description |
|--------------|-------------|
| Tot | Total marks |
| CM | Contested marks |
| In 50 | Marks inside forward 50 |
| Int Mks | Intercept marks |
| On lead | Marks on lead |

## Scoring

| Abbreviation | Description |
|--------------|-------------|
| Goals | Total goals |
| Gls Avg | Average goals per match |
| Beh | Behinds |
| Shots | Shots at goal |
| Goal Ass. | Goal assists |
| Acc | Accuracy (Goals / shots at goal) (minimum 10 shots at goal) |
| SI | Score involvements |
| SI % | Score involvements / Total team scores |
| Launch | Score launches |
| Off 1v1 | Offensive one-on-one contests |
| Win % | Percentage of offensive one-on-one contests won (minimum 10 contests) |

## Expected Scores (xScores)

| Abbreviation | Description |
|--------------|-------------|
| Shots | Total shots at goal |
| xSc | Average expected score per shot at goal, reflecting the average difficulty of shots taken |
| Rating | Average score above or below expected score per shot at goal |
| Shots (set) | Total shots at goal from set shots |
| xSc (Set) | Average expected score per shot at goal for set shots, reflecting the average difficulty of shots taken |
| Rating (Set) | Average score above or below expected score per shot at goal for set shots |
| Shots (Gen) | Total shots at goal from general play |
| xSc (Gen) | Average expected score per shot at goal for shots in general play, reflecting the average difficulty of shots taken |
| Rating (Gen) | Average score above or below expected score per shot at goal for shots in general play |

## Defence

| Abbreviation | Description |
|--------------|-------------|
| Def 1v1 | Defensive one-on-one contests |
| Loss % | Percentage of defensive one-on-one contests lost (minimum 10 contests) |
| Tack | Total tackles |
| Tack In 50 | Tackles inside forward 50 |
| Press. Acts | Pressure acts |
| Def. PA | Pressure acts in the defensive half |

## Ruck contests

| Abbreviation | Description |
|--------------|-------------|
| Ruck Cont | Ruck contests |
| RC % | Ruck contests / total team ruck contests |
| Hit Outs | Total hit outs |
| Win % | Hit outs / ruck contests (minimum 20 ruck contests) |
| To Adv | Hit outs to advantage |
| Adv % | Hit outs to advantage / total hit outs (minimum 20 hit outs) |
| Ruck HBG | Ruck hard ball gets. Taking possession of the ball directly out of the ruck |

## Other

| Abbreviation | Description |
|--------------|-------------|
| Kick In % | Kick-ins / Total kick-ins |
| KI Play On % | Play on percentage from kick-ins (minimum 10 kick-ins) |
| Bounce | Bounces |
| 1%s | One percenters |

## CSV Column Mappings

The following table maps the abbreviations to the actual column names in the CSV files:

| Abbreviation | CSV Column Name |
|--------------|----------------|
| Age | Age |
| TOG | TimeOnGround |
| Kick | Kicks |
| HB | Handballs |
| Dis. | Disposals |
| Eff % | DisposalEfficiency |
| Kick Eff % | KickingEfficiency |
| HB Eff % | HandballEfficiency |
| Kick % | KickPercentage |
| In 50s | Inside50s |
| Reb 50s | Rebound50s |
| Mtrs Gnd | MetresGained |
| Mtrs / D | MetresGainedPerDisposal |
| Clng | Clangers |
| D/Cg | DisposalsPerClanger |
| TO | Turnovers |
| D/TO | DisposalsPerTurnover |
| CP | ContestedPossessions |
| UP | UncontestedPossessions |
| Poss | TotalPossessions |
| CP% | ContestedPossessionRate |
| Int Poss | Intercepts |
| GB Gets | GroundBallGets |
| F50 GBG | GroundBallGetsForward50 |
| Hard Gets | HardBallGets |
| Loose Gets | LooseBallGets |
| Post Cl CP | PostClearanceContestedPossessions |
| Post Cl GBG | PostClearanceGroundBallGets |
| Gath HO | GathersFromHitout |
| Crumb | CrumbingPossessions |
| HB Rec | HandballReceives |
| CBA | CentreBounceAttendances |
| CBA % | CentreBounceAttendancePercentage |
| Ctr Clr | CentreClearances |
| CC / CBA | CentreClearanceRate |
| Stp Clr | StoppageClearances |
| Tot (Clearances) | TotalClearances |
| Tot (Marks) | Marks |
| CM | ContestedMarks |
| In 50 | MarksInside50 |
| Int Mks | InterceptMarks |
| On lead | MarksOnLead |
| Goals | Goals_Total |
| Gls Avg | Goals_Avg |
| Beh | Behinds |
| Shots | ShotsAtGoal |
| Goal Ass. | GoalAssists |
| Acc | GoalAccuracy |
| SI | ScoreInvolvements |
| SI % | ScoreInvolvementPercentage |
| Launch | ScoreLaunches |
| Off 1v1 | ContestOffensiveOneOnOnes |
| Win % | ContestOffensiveWinPercentage |
| Shots (xScores) | xScoreShots |
| xSc | xScorePerShot |
| Rating | xScoreRatingPerShot |
| Shots (set) | xScoreShots_Set |
| xSc (Set) | xScorePerShot_Set |
| Rating (Set) | xScoreRatingPerShot_Set |
| Shots (Gen) | xScoreShots_General |
| xSc (Gen) | xScorePerShot_General |
| Rating (Gen) | xScoreRatingPerShot_General |
| Def 1v1 | ContestDefensiveOneOnOnes |
| Loss % | ContestDefensiveLossPercentage |
| Tack | Tackles |
| Tack In 50 | TacklesInside50 |
| Press. Acts | PressureActs |
| Def. PA | PressureActsDefensiveHalf |
| Ruck Cont | RuckContests |
| RC % | RuckContestPercentage |
| Hit Outs | Hitouts |
| Win % (Ruck) | HitoutsWinPercentage |
| To Adv | HitoutsToAdvantage |
| Adv % | HitoutsToAdvantagePercentage |
| Ruck HBG | RuckHardBallGets |
| Kick In % | KickInPercentage |
| KI Play On % | KickInsPlayOnPercentage |
| Bounce | Bounces |
| 1%s | OnePercenters |
