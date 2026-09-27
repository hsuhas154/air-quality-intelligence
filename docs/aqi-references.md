# Sources for the AQI constants

Every number and every rule in `analysis/aqi.py` comes from one of the sources
below. This file exists so that a reviewer can check the calculator against
its authority instead of taking it on trust.

## The primary source

**`AQI-Calculator.xls`**, CPCB's own calculator, from
`https://cpcb.gov.in/upload/national-air-quality-index/AQI-Calculator.xls`.
Created January 2015, last saved October 2022. Sheet1 carries one formula per
pollutant in column D, a presence flag in column E, and the aggregation rule
in G11.

This is the operative authority for the arithmetic. Where a secondary source
disagrees with it, the spreadsheet wins, and one did; see the retraction
below.

`tests/test_aqi.py` holds `reference_sub_index`, a direct transcription of
those cell formulas, and checks this module against it across a dense grid for
every pollutant. The spreadsheet also ships with sample inputs and cached
results, reproduced in
`test_reproduces_the_worked_example_shipped_in_the_spreadsheet`.

### The anchor

Every cell formula anchors on the **upper limit of the previous AQI
category**, not on the displayed lower bound of the current band. From D10,
the PM2.5 cell:

```
IF(AND(C10>120,C10<=250), 300+(C10-120)*(100/130), ...)
```

The band is labelled 301 to 400 in the published table, and the formula
anchors at 300. Anchoring at 301 instead, as this project did, shifts every
sub-index by up to one point. The spreadsheet's own sample makes the error
visible: PM10 at 121 gives 114, and the old code gave 115.

Corroborated by the two worked examples on the Indian Economic Service's
Arthapedia page
(https://ies.gov.in/index.php/arthapedia/concept/national-air-quality-index):

| Reading | Published working | Published result | This code |
| --- | --- | --- | --- |
| PM2.5 150 ug/m3 | `300+[(150-120)*100/130]` | 323 | 323 |
| PM2.5 45 ug/m3 | `50+[(45-30)*50/30]` | 75 | 75 |

### The top band

The top band is **open, and the arithmetic has no ceiling**. Each formula ends
by continuing the last band's line:

| Pollutant | Top band formula | Slope source |
| --- | --- | --- |
| PM10 | `IF(C8>430, 400+(C8-430)*(100/80))` | the 350 to 430 band |
| PM2.5 | `IF(C10>250, 400+(C10-250)*(100/130))` | the 120 to 250 band |
| SO2 | `IF(C12>1600, 400+(C12-1600)*(100/800))` | the 800 to 1600 band |
| NO2 | `IF(C14>400, 400+(C14-400)*(100/120))` | the 280 to 400 band |
| CO | `IF(C16>34, 400+(C16-34)*(100/17))` | the 17 to 34 band |
| NH3 | `IF(C20>1800, 400+(C20-1800)*(100/600))` | the 1200 to 1800 band |

In every case the top-band slope is the previous band's slope. So the rule is
one sentence: **piecewise linear through the five published bands, extended
linearly past the last one.** There is no sixth band in this module for that
reason.

PM2.5 at 900 ug/m3 therefore gives 900, not 500. The spreadsheet's own
category label for Severe is `(>401)`, open-ended, not `401-500`.

### The ceiling at 500 is a reporting convention

The category table caps Severe at 401 to 500, and CPCB does not publish above
500. Gufran Beig, founder-director of the government's SAFAR system, quoted in
*The Tribune*
(https://www.tribuneindia.com/news/haryana/who-approved-apps-show-aqi-of-1000-govt-caps-it-at-500):

> CPCB puts a cap on the AQI, limiting it to 500. This means that even when
> concentrations exceed 500 micrograms per cubic metre, the AQI cannot go
> higher.

That is a statement about what gets published, not about the arithmetic, and
the spreadsheet proves the two are different. So this module keeps them
separate: `SubIndex.value` is the uncapped official arithmetic,
`SubIndex.reported_value` applies `AQI_REPORTING_MAXIMUM`, and
`SubIndex.extrapolated` marks a reading that sits past the highest published
breakpoint. `daily_aqi.extrapolated` records it per station-day so the
frequency can be counted in SQL rather than assumed to be zero.

### Retraction: the closed top bands

An earlier version of this file, and of the calculator, gave PM2.5 a closed
top band of 250 to 350 and PM10 one of 430 to 500, citing Table 3.1 of the
Ministry of Earth Sciences EMRC Standard Operating Procedure
(https://mausam.imd.gov.in/imd_latest/contents/pdf/emrc_sop.pdf), which prints
"251 to 350" and "431 to 500" where other tables print "250+" and "430+".

CPCB's spreadsheet contradicts that directly: both bands are open and both
extrapolate. The SOP figures are withdrawn. They were the best available
reading before the spreadsheet could be opened, and they were wrong.

### The O3 transcription error in the official sheet

Two cells for O3 do not follow the pattern every other pollutant follows.

The 208 to 748 band divides by 539 rather than by its actual width of 540, so
at 748 it returns 400.19 instead of 400. That is a rounding artefact and is
harmless at integer precision.

The top band is a real error:

```
IF(C18>748, 400+(C18-400)*(100/539))
```

It subtracts 400, the AQI anchor, where every other pollutant subtracts the
band's own concentration breakpoint. Taken literally, O3 at 748 gives 400 and
O3 at 749 gives 465: a 65-point jump across one unit of ozone.

This module uses the consistent form, `400+(C-748)*(100/540)`.
`test_o3_top_band_is_continuous` pins it and will fail if anyone transcribes
the sheet's version back in.

## Minimum data requirements

### Three pollutants, one of which must be particulate matter

G11, the aggregation cell:

```
IF(AND(OR((E8)=1,(E10)=1),(E8+E10+E12+E14+E16+E18+E20)>=3),
   MAX(D8,D10,D12,D14,D16,D18,D20),
   "Atleast 3 inputs*")
```

E8 is PM10 present and E10 is PM2.5 present, so the `OR` is the
particulate-matter requirement and the sum is the count. Cell A21 states it in
words:

> Concentrations of minimum three pollutants are required; one of them should
> be PM10 or PM2.5

Note that the count includes **NH3**, which is why `BREAKPOINTS` carries it
even though this project does not ingest it.

Implemented as `has_sufficient_pollutants` and the `enforce_minimum_pollutants`
gate in `analysis/aqi.py`.

### A reading of zero counts as missing

The presence flag, from E8:

```
IF(OR(ISTEXT(C8),C8<=0),0,1)
```

A concentration of zero or below is treated as **missing**, not as a
measurement of clean air. So a day carrying PM2.5, NO2 and a zero for SO2 has
two pollutants, not three, and gets no AQI. Implemented as `is_present`, which
also drops absent pollutants from the returned sub-indices.

### Sixteen hours

Arthapedia, same page as above:

> a minimum of 16 hours' data is considered necessary for calculating sub
> index

Implemented as `MINIMUM_HOURS_FOR_DAILY_MEAN = 16` in `analysis/daily_aqi.py`,
applied to the 24-hourly pollutants per calendar day.

Before these rules, the daily table published an AQI of 2 for Delhi and 7 for
Bengaluru. Both came from days where one pollutant, sometimes from a single
hourly reading, was averaged on its own.

The 16-hour figure is published for the 24-hour averaging period. No
equivalent figure was found for the 8-hour pollutants, so this project
requires a **complete** 8-hour window for CO and O3, which is stricter than
any published minimum and consistent with `analysis/temporal.py`, the module
that builds the model target. That choice is the project's, not CPCB's.

## Averaging periods

24-hour mean for PM2.5, PM10, NO2, SO2 and NH3; maximum 8-hour mean for CO and
O3. Source: column B of the spreadsheet, which labels each pollutant row with
its averaging period. Implemented in `analysis/temporal.py`
(`AVERAGING_HOURS`) and mirrored by the long and short term pollutant tuples
in `analysis/daily_aqi.py`.

## Aggregation

`Final AQI = maximum of all valid sub-indices`, per G11 above and per the
methodology chapter of the National Air Quality Index Final Report
(CUPS/82/2014-15, CPCB with IIT Kanpur, October 2014), which also gives the
sub-index formula as
`Sub-index(Ip) = [(IHI - ILO) / (BHI - BLO)] x (Cp - BLO) + ILO`.

## Plausibility bounds

These are not CPCB's, and `transform/validation.py` says so. They exist to
catch a mislabelled provider unit, not to bound reality, and each is derived
from a stated anchor:

- **Upper:** ten times the highest concentration the published breakpoint
  table names, read from `BREAKPOINTS` so the two cannot drift apart. The
  bound clears the worst documented episode, an hourly mean PM2.5 of 1341
  ug/m3 over northwest India in November 2022 (Sci Rep, PMC10425363), against
  a PM2.5 bound of 2500. It stays far below a mislabelled value: the unit
  confusions in play are factors of 1000, so a typical Delhi PM2.5 of 100
  arrives as 100000.
- **Lower:** zero for everything except CO. CO has a global atmospheric floor;
  Northern Hemisphere background mixing ratios have not fallen below roughly
  90 ppb even in preindustrial ice-core reconstructions (Clim Past 18:631,
  2022), about 0.103 mg/m3 at this project's conversion factor. A city cannot
  be cleaner than the background it sits in, so an ambient CO reading well
  under that is a unit error rather than clean air.

## Sources not used, and why

- `airquality.cpcb.gov.in`, `cpcb.nic.in` and `clip.cpcb.gov.in` refuse
  automated fetches, so their copies of the table were not read directly.
- The Ministry of Earth Sciences SOP's closed top bands are withdrawn; see the
  retraction above.
- The CPCB breakpoint table reproduced in *IJERPH* 15(7):1460 prints PM10 as
  "431 to 550", O3 as "375 to 450" and NO2 as "214 to 750" in the Severe row.
  Those disagree with CPCB's own calculator and are not used.
