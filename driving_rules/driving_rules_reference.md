# Driving Rules Reference — CausalDriveBench

Hierarchical driving rules for VLA evaluation in Singapore (SG) and US-California (US_CA) contexts.  
Source files: `taxonomy.yaml`, `generic_rules.yaml`, `sg_specific_rules.yaml`, `us_ca_specific_rules.yaml`

---

## Priority System

| Priority | Label | When It Applies |
|---|---|---|
| **0** | ABSOLUTE_OVERRIDE | Emergency vehicles, DUI prohibition, imminent collision |
| **1** | SAFETY_CRITICAL_STOP | Red lights, stop signs, railroad crossings, school bus, seatbelts |
| **2** | RIGHT_OF_WAY_AND_YIELD | Yield signs, pedestrian ROW, left-turn yield, roundabouts |
| **3** | SPEED_REGULATION | Posted limits, zone limits, weather-based speed reduction |
| **4** | LANE_DISCIPLINE | Lane changes, passing, HOV/bus/bike lane compliance |
| **5** | TURNING_AND_MANEUVERING | Turn execution, U-turns, merging, reversing |
| **6** | PARKING_AND_STOPPING | Parking zones, load/unload, hill parking |
| **7** | VEHICLE_EQUIPMENT_AND_LOAD | Headlights, cargo, windshield, vehicle condition |
| **8** | ADVISORY_AND_GUIDANCE | Best-practice recommendations |

> **Conflict rule:** When two rules apply simultaneously, the **lower** priority number wins.  
> Ties → take the **most conservative** action (stop/yield rather than proceed).

---

## Hierarchical Taxonomy

```
TRAFFIC_CONTROL
├── TRAFFIC_SIGNALS         (red/amber/green, arrows, flashing, dark signals)
├── TRAFFIC_SIGNS           (stop, yield, prohibitory, warning, mandatory)
├── ROAD_MARKINGS           (stop/give-way lines, lane lines, arrows)
└── DYNAMIC_SIGNS           (EMAS, variable speed, electronic message boards)

RIGHT_OF_WAY
├── EMERGENCY_PRIORITY      (ambulance, fire, police — absolute P0)
├── PEDESTRIAN_ROW          (crosswalks, guide dogs, vulnerable users)
├── INTERSECTION_ROW        (first-arrival, yield-to-right, T-junction)
└── SPECIAL_ROW             (roundabouts, mountain roads, merging)

SPEED_REGULATION
├── POSTED_LIMITS           (speed limit signs)
├── ZONE_LIMITS             (school, construction, residential, alley)
└── CONDITION_BASED         (rain, fog, ice, night, slippery roads)

LANE_DISCIPLINE
├── LANE_USAGE              (left-keep, overtake lanes, lane selection)
├── LANE_CHANGES            (signal, blind spot, solid line prohibition)
├── PASSING                 (overtake conditions, hill/curve prohibition)
└── SPECIAL_LANES           (HOV, bus, bike, turn, center-turn lanes)

TURNING_AND_MANEUVERING
├── TURNS                   (right/left turn procedure, lane positioning)
├── U_TURNS                 (permitted locations, prohibited locations)
├── MERGING                 (highway entry/exit, yield to highway)
└── REVERSING               (when/where prohibited, expressways)

PEDESTRIAN_SAFETY
├── CROSSWALK_RULES         (marked/unmarked crosswalks, stopping before)
├── VULNERABLE_USERS        (elderly, children, disabled, white cane)
└── SCHOOL_ZONES            (speed near schools, school bus red lights)

PARKING_AND_STOPPING
├── PROHIBITED_PARKING      (crosswalk, fire hydrant, no-stopping zones)
├── RESTRICTED_PARKING      (time-limited, conditional zones)
├── LOADING_ZONES           (pickup/dropoff only areas)
└── CURB_COLORS [US_CA]     (white/green/yellow/red/blue curb rules)

VEHICLE_SAFETY
├── OCCUPANT_RESTRAINTS     (seatbelts, child seats)
├── IMPAIRED_DRIVING        (DUI, BAC limits — P0)
├── DISTRACTED_DRIVING      (phones, attention)
├── HEADLIGHTS              (low-beam, high-beam, fog)
└── VEHICLE_CONDITION       (load, cargo, windshield, dimensions)

SPECIAL_SITUATIONS
├── RAILROAD_CROSSINGS      (stop, never on tracks, never under gate)
├── SCHOOL_BUSES            (red flash = mandatory stop both directions)
├── EMERGENCY_VEHICLES      (pull right, no close-following)
├── WORK_ZONES              (reduced speed, flagger authority)
├── EXPRESSWAYS             (no reverse, vehicle restrictions)
└── TUNNELS                 (dangerous goods ban)

ROAD_USER_INTERACTION
├── LARGE_VEHICLES          (blind spots, braking distance, no cut-in)
├── MOTORCYCLES             (full lane, 3s following, lane splitting [CA])
├── BICYCLES                (passing clearance, bike lane, night lights)
└── SLOW_MOVING             (orange triangle, speed differential)
```

---

## Generic Driving Rules

Rules applicable in **both Singapore and California** (and most jurisdictions).

### Traffic Control — Traffic Signals

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `STOP_AT_RED_LIGHT` | Stop at Red Light | **1** | Red circular signal visible | Full stop before stop line. Wait for green. |
| `STOP_AT_RED_ARROW` | Stop at Red Arrow | **1** | Red arrow on dedicated turn signal | Full stop. No turn until green arrow. |
| `STOP_AT_FLASHING_RED` | Treat Flashing Red as Stop Sign | **1** | Signal flashing red | Full stop. Yield to cross traffic. |
| `DARK_SIGNAL_FOUR_WAY_STOP` | Non-Working Signal = 4-Way Stop | **1** | Signal head completely unlit | Treat as all-way stop. |
| `STOP_OR_CLEAR_AMBER_LIGHT` | Stop or Clear on Amber | **1** | Amber signal appears | Stop if safe; clear if already committed. |
| `CAUTION_AT_AMBER_ARROW` | Complete Turn or Stop on Amber Arrow | **2** | Amber arrow on turn signal | If mid-turn: complete. If waiting: stop. |
| `CAUTION_AT_FLASHING_AMBER` | Caution at Flashing Amber | **3** | Flashing amber signal | Slow, scan, proceed when safe. |
| `PROCEED_GREEN_YIELD_TO_CONFLICTS` | Green — Yield to Pedestrians & Clearing Vehicles | **3** | Green circular signal turns on | Check pedestrians and clearing vehicles first. |
| `PROCEED_ON_GREEN_ARROW` | Protected Turn on Green Arrow | **3** | Green arrow on turn signal | Proceed in arrow direction, yield to pedestrians. |

### Traffic Control — Traffic Signs

| Rule ID | Name | Priority | Key Sign Name(s) | Required Action |
|---|---|---|---|---|
| `FULL_STOP_AT_STOP_SIGN` | Full Stop at STOP Sign | **1** | `SIGN_STOP` | Zero-velocity stop before line. Check all directions. |
| `YIELD_AT_YIELD_SIGN` | Yield at YIELD Sign | **2** | `SIGN_YIELD` | Slow, yield to cross traffic. Stop if needed. |
| `OBEY_NO_ENTRY_SIGN` | Do Not Enter | **1** | `SIGN_NO_ENTRY` | Do not enter. Find alternate route. |
| `OBEY_WRONG_WAY_SIGN` | Wrong Way — Do Not Proceed | **1** | `SIGN_WRONG_WAY` | Stop, reverse, turn around. Activate hazards. |
| `OBEY_SPEED_LIMIT_SIGN` | Observe Posted Speed Limit | **3** | `SIGN_SPEED_LIMIT` | Travel at or below posted limit. |
| `OBEY_NO_OVERTAKING_SIGN` | No Overtaking Zone | **3** | `SIGN_NO_OVERTAKING` | Do not pass. Stay behind vehicle ahead. |
| `OBEY_MANDATORY_DIRECTION_SIGN` | Obey Mandatory Direction | **2** | `SIGN_TURN_RIGHT_ONLY` `SIGN_AHEAD_ONLY` `SIGN_KEEP_LEFT` | Follow indicated direction exactly. |
| `SLOW_AT_WARNING_SIGNS` | Reduce Speed at Warning Signs | **3** | `SIGN_WARNING_*` | Reduce speed, scan for warned hazard. |
| `OBEY_NO_U_TURN_SIGN` | No U-Turn Where Posted | **2** | `SIGN_NO_U_TURN` | Do not U-turn. Find next legal location. |
| `RAILROAD_CROSSING_ADVANCE_SIGN` | Prepare to Stop at Railroad Crossing | **1** | `SIGN_RAILROAD_CROSSBUCK` | Reduce speed. Look and listen both ways. |

### Traffic Control — Road Markings

| Rule ID | Name | Priority | Marking Name(s) | Required Action |
|---|---|---|---|---|
| `STOP_AT_STOP_LINE` | Stop Before Stop Line | **1** | `MARKING_STOP_LINE` | Stop with front bumper behind line. |
| `YIELD_AT_BROKEN_GIVE_WAY_LINE` | Yield at Give-Way Line | **2** | `MARKING_GIVE_WAY_LINE` | Slow, yield to priority traffic. |
| `NO_CROSS_SOLID_CENTER_LINE` | Do Not Cross Solid Center Line | **2** | `MARKING_SOLID_YELLOW_CENTER` `MARKING_DOUBLE_SOLID_YELLOW` | Do not cross. Wait for broken section. |
| `PASSING_ON_BROKEN_CENTER_LINE` | May Pass on Broken Center Line | **4** | `MARKING_BROKEN_YELLOW_CENTER` | Pass only if safe, adequate sight distance. |
| `LANE_ARROW_COMPLIANCE` | Follow Lane Arrow Markings | **4** | `MARKING_LANE_ARROW_*` | Use only directions indicated in your lane. |

### Right of Way

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `EMERGENCY_VEHICLE_PRIORITY` | Yield to Emergency Vehicle | **0** | Siren+lights active, any direction | Pull right, stop. Wait until passed. |
| `PEDESTRIAN_CROSSWALK_ROW` | Yield to Pedestrians at Crosswalk | **1** | Pedestrian in or entering crosswalk | Stop before crosswalk. Wait for full clearance. |
| `GUIDE_DOG_ABSOLUTE_ROW` | Yield to Guide Dog / White Cane Users | **1** | White cane or guide dog visible | Stop immediately. Do NOT honk. |
| `INTERSECTION_FIRST_ARRIVAL_ROW` | First Arrival Has Priority | **2** | Uncontrolled intersection | First to arrive proceeds. |
| `YIELD_TO_VEHICLE_ON_RIGHT` | Yield to Vehicle on Right (Simultaneous) | **2** | Simultaneous arrival at intersection | Vehicle on left yields to vehicle on right. |
| `THROUGH_TRAFFIC_T_INTERSECTION` | Through Road Priority at T-Junction | **2** | Entering T-junction from dead-end road | Yield to through-road traffic. |
| `LEFT_TURN_YIELD_TO_ONCOMING` | Left Turn Yields to Oncoming Traffic | **2** | Turning left without protected arrow | Wait for oncoming to clear, then turn. |
| `ROUNDABOUT_YIELD_TO_INSIDE` | Yield to Traffic in Roundabout | **2** | Entering roundabout | Yield to all circulating traffic. |
| `UPHILL_PRIORITY_NARROW_ROAD` | Uphill Vehicle Has Priority | **3** | Narrow road, two vehicles on hill | Downhill vehicle reverses/yields. |
| `MERGE_YIELD_TO_HIGHWAY_TRAFFIC` | Merging Vehicle Yields to Highway | **2** | On-ramp or slip road merge | Match speed, merge into safe gap. |

### Speed Regulation

| Rule ID | Name | Priority | Trigger Condition | Required Action |
|---|---|---|---|---|
| `SCHOOL_ZONE_REDUCED_SPEED` | Reduced Speed in School Zone | **2** | School zone sign, children present | Reduce to posted zone limit. |
| `ADVERSE_WEATHER_SPEED_REDUCTION` | Reduce Speed in Adverse Weather | **2** | Rain, fog, snow, ice | Reduce speed, increase following distance. |
| `REDUCED_VISIBILITY_SPEED` | Reduce Speed in Low Visibility | **2** | Night, fog, tunnel entry | Match speed to sight distance. |
| `CONSTRUCTION_ZONE_SPEED_LIMIT` | Work Zone Speed Limit | **2** | Construction zone signage | Observe reduced posted limit. Obey flaggers. |
| `SLIPPERY_ROAD_SPEED_ADJUSTMENT` | Reduce Speed on Slippery Roads | **2** | Wet/oily/gravelly surface | Reduce speed, brake gently, increase gap. |
| `SAFE_FOLLOWING_DISTANCE_3S` | 3-Second Following Distance | **3** | Following any vehicle at speed | Maintain 3s gap. Increase in adverse conditions. |

### Lane Discipline

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `SIGNAL_BEFORE_LANE_CHANGE` | Signal Before Lane Change | **3** | Before every lane change | Activate signal, check mirrors + blind spot. |
| `CHECK_BLINDSPOT_BEFORE_LANE_CHANGE` | Check Blind Spots | **3** | Before every lane change | Turn head to verify blind spot is clear. |
| `STAY_IN_LANE_NO_WEAVING` | Stay in Lane — No Weaving | **4** | Any multi-lane road | Stay centered, change lanes only when needed. |
| `NO_PASSING_ON_HILLS_CURVES` | No Passing on Hills/Curves | **2** | Limited sight distance ahead | Do not attempt overtake. |
| `NO_PASSING_NEAR_HAZARDOUS_AREAS` | No Passing Near Intersections/Crossings | **2** | Within 100 ft of hazard | Defer overtake to safe section. |
| `ALLOW_PASSING_VEHICLE_TO_COMPLETE` | Allow Passing Vehicle to Finish | **4** | Vehicle is overtaking ego vehicle | Maintain speed/lane. Do not accelerate. |
| `BIKE_LANE_RESTRICTION` | Do Not Drive in Bike Lane | **4** | Bike lane present | Avoid bike lane. Yield to cyclists when crossing. |
| `DO_NOT_BLOCK_INTERSECTION` | Do Not Enter Blocked Intersection | **2** | Green light but no clearance ahead | Wait at stop line until intersection can be cleared. |

### Turning and Maneuvering

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `SIGNAL_BEFORE_TURNING` | Signal Before Turning | **3** | Before every turn | Activate turn signal ≥3 seconds / 100 ft before. |
| `RIGHT_TURN_FROM_RIGHT_LANE` | Right Turn from Right Lane | **4** | Any right turn | Position in rightmost lane. Complete into rightmost lane. |
| `LEFT_TURN_FROM_CENTER_LANE` | Left Turn from Left/Center Lane | **4** | Any left turn | Position in leftmost or turn lane. Complete into leftmost lane. |
| `KEEP_WHEELS_STRAIGHT_BEFORE_TURN` | Wheels Straight While Waiting to Turn Left | **4** | Waiting in intersection to turn left | Keep wheels straight until executing turn. |
| `COMPLETE_TURN_IN_CORRECT_LANE` | Complete Turn into Correct Lane | **4** | Any turn at multi-lane intersection | Right turn → right lane. Left turn → left lane. |
| `NO_U_TURN_LOW_VISIBILITY` | No U-Turn in Low Visibility | **2** | Sight distance < 500 ft / 150 m | Do not U-turn. Find clear location. |
| `NO_U_TURN_AT_RAILROAD_CROSSING` | No U-Turn at Railroad Crossing | **1** | Near railroad tracks | Never U-turn at crossing. |
| `NO_REVERSE_ON_EXPRESSWAY` | No Reversing on Expressway | **1** | Any high-speed divided road | Continue to next exit. Never reverse. |

### Pedestrian Safety

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `STOP_FOR_PEDESTRIAN_AT_CROSSING` | Stop for Pedestrian at Any Crosswalk | **1** | Pedestrian at/in crosswalk | Stop before crosswalk. Wait for full clearance. |
| `DO_NOT_STOP_IN_CROSSWALK` | Do Not Stop in Crosswalk | **2** | Stopping at intersection | Stop before crosswalk, not in it. |
| `DO_NOT_PASS_STOPPED_AT_CROSSWALK` | Do Not Pass Vehicle Stopped at Crosswalk | **2** | Vehicle ahead stopped at crosswalk | Stop behind it. Hidden pedestrian may be crossing. |
| `YIELD_EXTRA_TO_VULNERABLE_PEDESTRIANS` | Extra Caution for Vulnerable Pedestrians | **2** | Elderly, children, wheelchair users near road | Slow earlier, allow more crossing time. |
| `NO_HONK_AT_BLIND_PEDESTRIANS` | No Horn at Visually Impaired Pedestrians | **4** | White cane / guide dog user present | Wait silently. Do not honk. |
| `NO_DRIVE_OVER_RAISED_CROSSING` | No Overtake/U-Turn Near Raised Crossing | **2** | Raised or table-top crossing | Slow, no overtake, no U-turn, yield to pedestrians. |

### Parking and Stopping

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `NO_PARKING_IN_CROSSWALK` | No Parking in Crosswalk | **4** | Any crosswalk location | Never stop/park in crosswalk. |
| `NO_PARKING_NEAR_FIRE_HYDRANT` | No Parking Near Fire Hydrant | **4** | Near any fire hydrant | Do not park within required clearance distance. |
| `NO_PARKING_IN_PROHIBITED_ZONES` | No Parking Where Prohibited | **4** | Any no-parking sign or marking | Do not park, stand, or stop. |
| `HILL_PARKING_WHEEL_POSITION` | Correct Wheel Position on Hill | **6** | Parking on incline | Turn wheels toward/away from curb per slope direction. |
| `DOUBLE_PARKING_PROHIBITED` | No Double Parking | **4** | Adjacent to parked vehicle in traffic lane | Find legal space. |

### Vehicle Safety

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `SEATBELT_MANDATORY_ALL` | Seatbelt Mandatory | **1** | Any time vehicle is moving | All occupants buckled before moving. |
| `CHILD_RESTRAINT_MANDATORY` | Child Restraint Mandatory | **1** | Child passenger present | Age-appropriate child restraint. |
| `DUI_ABSOLUTE_PROHIBITION` | No Driving Under the Influence | **0** | Always | Never drive while impaired. Pull over if impaired. |
| `NO_DISTRACTED_DRIVING` | No Distracted Driving | **1** | Always while driving | Eyes on road, hands on wheel, mind on driving. |
| `HEADLIGHT_USE_IN_LOW_VISIBILITY` | Use Headlights in Low Visibility | **3** | Night, rain, fog, tunnels | Low-beam on. Do not use parking lights only. |
| `HIGH_BEAM_DIM_FOR_ONCOMING_TRAFFIC` | Dim High Beams for Oncoming Traffic | **3** | Oncoming vehicle within ~500 ft | Switch to low beam. Return to high after passing. |
| `NO_OVERLOADED_VEHICLE` | No Overloaded or Obstructed Vehicle | **3** | Before every trip | Secure cargo, clear windows and mirrors. |

### Special Situations

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `RAILROAD_CROSSING_MANDATORY_STOP` | Stop at Railroad Crossing When Activated | **1** | Flashing lights, gate, or audible bell | Stop ≥15 ft from tracks. Wait for gate + lights to clear. |
| `NEVER_STOP_ON_TRACKS` | Never Stop on Railroad Tracks | **1** | Approaching any crossing | Do not proceed unless far side is clear. |
| `NEVER_CROSS_LOWERED_GATE` | Never Cross Under Lowered Gate | **1** | Gate arm is down | Stop. Never drive under gate. |
| `SCHOOL_BUS_STOP_RED_LIGHTS` | Stop for School Bus Red Lights | **1** | School bus flashing red, STOP arm out | Stop both directions. Wait until lights stop. |
| `EMERGENCY_VEHICLE_PULL_RIGHT_STOP` | Pull Right and Stop for Emergency Vehicle | **0** | Emergency vehicle approaching with lights/siren | Move to right edge, stop, wait until passed. |
| `NO_FOLLOW_EMERGENCY_VEHICLE_CLOSE` | No Tailgating Emergency Vehicle | **2** | Emergency vehicle ahead with lights/siren | Maintain ≥300 ft / 100 m. |
| `WORK_ZONE_FOLLOW_FLAGGER` | Follow Flagger Instructions | **2** | Flagger visible in work zone | Obey STOP/SLOW paddle. Do not pass unless directed. |

### Road User Interaction

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `LARGE_VEHICLE_BLIND_SPOT_AVOIDANCE` | Avoid Truck/Bus Blind Spots | **2** | Near any large vehicle | Pass quickly. Do not linger in blind zones. |
| `NO_SUDDEN_STOP_BEFORE_TRUCKS` | No Sudden Stop Before Large Vehicles | **2** | Merging or braking in front of truck/bus | Allow full vehicle length + 50% before cutting in. |
| `MOTORCYCLE_FULL_LANE_RESPECT` | Give Motorcycles Full Lane Width | **3** | Motorcycle in adjacent or same lane | Full lane gap. Check blind spots. |
| `BICYCLE_PASSING_CLEARANCE` | Safe Lateral Clearance Passing Cyclists | **2** | Overtaking cyclist on any road | Move left, maintain adequate clearance. |
| `PASS_SAFETY_ZONE_SLOWLY` | Slow When Passing Bus/Tram Safety Zone | **3** | Bus/tram stopped at in-road safety zone | Reduce to ≤10 mph / 15 km/h. Watch for passengers. |
| `SLOW_MOVING_VEHICLE_AWARENESS` | Alert to Slow-Moving Vehicles | **4** | Orange triangle reflector vehicle ahead | Reduce speed and following distance. Pass safely. |

---

## Singapore-Specific Rules

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `SG_KEEP_LEFT_OVERTAKE_RIGHT` | Keep Left — Right Lanes Overtaking Only | **4** | Any multi-lane road | Stay left. Use right only to overtake; return left. |
| `SG_PART_DAY_BUS_LANE_RESTRICTION` | No Driving in Part-Day Bus Lane (Peak Hours) | **4** | Weekdays 7:30–9:30 AM, 5:00–8:00 PM | Do not enter red bus lane during restricted hours. |
| `SG_FULL_DAY_BUS_LANE_RESTRICTION` | No Driving in Full-Day Bus Lane | **4** | Weekdays 7:30 AM–11:00 PM | Do not enter full-day bus lane during restricted hours. |
| `SG_BUS_PRIORITY_BOX_YIELD` | Yield to Buses Exiting Bus Priority Box | **2** | Yellow BUS PRIORITY box ahead | Do not enter box. Yield to exiting bus. |
| `SG_NO_REMAIN_IN_BUS_PRIORITY_BOX` | Do Not Stop in Bus Priority Box | **3** | Bus Priority Box with congestion ahead | Wait before box. Enter only when clear to exit. |
| `SG_ZEBRA_CROSSING_FULL_STOP` | Stop for Pedestrians at Zebra Crossing | **1** | Pedestrian at zebra crossing | Full stop before white line. Wait for full clearance. |
| `SG_NO_OVERTAKE_NEAR_ZEBRA_CROSSING` | No Overtaking on Approach to Zebra Crossing | **2** | Within zigzag approach zone | No overtake. Slow and prepare to stop. |
| `SG_NO_U_TURN_NEAR_ZEBRA_CROSSING` | No U-Turn Near Zebra/Raised Crossing | **2** | Within approach zone of crossing | No U-turn. Find next legal location. |
| `SG_VISUALLY_HANDICAPPED_YIELD` | Yield Absolutely to White Cane Pedestrians | **1** | White cane pedestrian anywhere near road | Stop, yield, do not honk. |
| `SG_CHILDREN_ZONE_REDUCE_SPEED` | Reduced Speed in Children/School/Silver Zones | **2** | Within marked zone | Reduce to posted zone limit. Watch for pedestrians. |
| `SG_WHITE_ZIGZAG_NO_STOPPING` | No Stopping on White Zigzag Lines | **3** | White zigzag zone near crossings | Do not park, wait, or stop (signals excepted). |
| `SG_YELLOW_ZIGZAG_NO_STOPPING` | No Stopping on Yellow Zigzag Lines | **3** | Yellow zigzag road markings | No stopping at any time. Incurs demerit + fine. |
| `SG_DOUBLE_YELLOW_ZIGZAG_NO_STOPPING` | No Stopping — Double Yellow Zigzag | **2** | Double yellow zigzag anywhere | Absolute no-stopping. No exceptions. |
| `SG_UNBROKEN_YELLOW_LINE_PARKING_HOURS` | No Parking on Solid Yellow 7 AM–7 PM Weekdays | **5** | Weekdays 7:00 AM–7:00 PM | No parking during restricted hours. |
| `SG_DOUBLE_YELLOW_LINE_NO_PARK_ANYTIME` | No Parking on Double Yellow Lines | **4** | Double yellow curb lines, at any time | No parking. Immediate pickup only (driver stays). |
| `SG_NO_PARK_AT_BUS_STOP_9M` | No Parking Within 9 m of Bus Stop Bay | **4** | Within 9 m of bus stop | Do not park. |
| `SG_RAG_DEDICATED_TURN_SIGNAL` | Wait for RAG Green Arrow Before Turning | **2** | RAG dedicated turn signal present | Wait for green arrow. Standard green = no turn. |
| `SG_RED_LIGHT_CAMERA_COMPLIANCE` | Stop on Amber — RLC Enforced Junctions | **1** | Approaching any SG junction on amber | Stop on amber if safe. Do not beat signal. |
| `SG_EXPRESSWAY_NO_REVERSE` | No Reversing or U-Turning on Expressway | **1** | On any SG expressway | Continue to next exit. Never reverse. |
| `SG_EXPRESSWAY_VEHICLE_RESTRICTIONS` | Certain Vehicles Prohibited on Expressways | **2** | Approaching expressway with restriction signs | Check sign. Do not enter if your vehicle type is restricted. |
| `SG_TUNNEL_NO_DANGEROUS_GOODS` | No Dangerous Goods in Tunnels | **2** | Carrying hazardous cargo, tunnel ahead | Use surface route. Do not enter tunnel. |
| `SG_HORN_ONLY_TO_PREVENT_ACCIDENT` | Horn Only to Prevent Accident | **4** | Always while driving in SG | Sound horn only for immediate danger warning. |
| `SG_VEHICLE_DIMENSION_COMPLIANCE` | Oversized Loads Require Escort | **4** | Vehicle/load exceeds posted limits | Arrange approved escort and route. |
| `SG_EMAS_DYNAMIC_SIGN_COMPLIANCE` | Obey EMAS Electronic Message Signs | **2** | EMAS sign on expressway gantry | Obey immediately. Variable speed overrides posted. |
| `SG_TOW_AWAY_ZONE_COMPLIANCE` | No Parking in Tow-Away or Wheel-Clamp Zones | **3** | Tow-away or wheel-clamp zone signage | Do not park. |

---

## US California-Specific Rules

### Speed Limits (CA)

| Rule ID | Name | Priority | Default/Posted Speed | Trigger |
|---|---|---|---|---|
| `CA_SPEED_RESIDENTIAL_25MPH` | 25 mph in Business/Residential Districts | **3** | 25 mph | Residential/business area, no higher limit posted |
| `CA_SPEED_SCHOOL_ZONE_25MPH` | 25 mph in School Zone | **2** | 25 mph (some zones 15) | Within 500 ft of school when children present |
| `CA_SPEED_BLIND_INTERSECTION_15MPH` | 15 mph at Blind Intersections | **3** | 15 mph | Uncontrolled intersection, view blocked within 100 ft |
| `CA_SPEED_ALLEY_15MPH` | 15 mph in Alleys | **3** | 15 mph | Any alley |
| `CA_SPEED_RAILROAD_UNCONTROLLED_15MPH` | 15 mph at Uncontrolled RR Crossing | **2** | 15 mph | Within 100 ft of uncontrolled crossing, tracks not visible 400 ft |
| `CA_SPEED_STREETCAR_SAFETY_ZONE_10MPH` | 10 mph Passing Bus/Tram Safety Zone | **3** | 10 mph | Bus/tram safety zone visible with stopped vehicle |

### Lane Discipline (CA)

| Rule ID | Name | Priority | Key Sign/Marking | Required Action |
|---|---|---|---|---|
| `CA_HOV_MINIMUM_OCCUPANCY` | HOV Lane: Minimum Occupancy | **3** | `SIGN_HOV_LANE` `MARKING_DIAMOND_HOV` | Meet occupancy requirement or have exempt vehicle. |
| `CA_HOV_ENTER_DESIGNATED_ONLY` | Enter HOV Lane at Dashed Line Breaks Only | **3** | `MARKING_DOUBLE_SOLID_WHITE` | Only enter/exit at broken white line sections. |
| `CA_HOV_LOW_EMISSION_EXEMPT` | Low/Zero Emission Vehicle HOV Exempt | **4** | `SIGN_CLEAN_AIR_VEHICLE` | Valid DMV decal required. |
| `CA_TWO_WAY_LEFT_TURN_LANE` | Enter TWLTL Within 200 Feet Only | **4** | `MARKING_TWO_WAY_LEFT_TURN_LANE` | Enter only within 200 ft of intended turn. |

### Turning and Signals (CA)

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `CA_RIGHT_TURN_ON_RED` | Right Turn on Red (After Stop, If No Sign) | **3** | Red circular (not arrow), no NTOR sign | Full stop first. Yield to all. Then turn if clear. |
| `CA_LEFT_TURN_ONE_WAY_TO_ONE_WAY` | Left Turn on Red (One-Way → One-Way) | **3** | One-way → one-way, red circular, no NTOR | Full stop. Yield. Turn if clear. |
| `CA_SIGNAL_100FT_BEFORE_TURN` | Signal ≥100 ft Before Turn | **3** | Any turn or lane change | Activate signal ≥100 ft (urban) / 400 ft (freeway). |
| `CA_U_TURN_BUSINESS_DISTRICT` | U-Turn Only at Intersections in Business Dist. | **3** | Business district, intending U-turn | U-turn only at intersections or divided openings. |
| `CA_U_TURN_RESIDENTIAL_200FT` | U-Turn Legal in Residential if 200 ft Clear | **3** | Residential area, 200 ft clear both ways | Signal, verify clearance, complete quickly. |

### Parking — Colored Curbs (CA)

| Rule ID | Curb Color | Priority | Meaning | Permitted Action |
|---|---|---|---|---|
| `CA_COLORED_CURB_WHITE` | White | **5** | Passenger load/unload only | Stop briefly, driver stays, move on. |
| `CA_COLORED_CURB_GREEN` | Green | **5** | Limited time parking | Park within posted time (default 10 min). |
| `CA_COLORED_CURB_YELLOW` | Yellow | **5** | Freight/passenger load-unload during business hours | Load/unload only, stay with vehicle. |
| `CA_COLORED_CURB_RED` | Red | **4** | No stopping/standing/parking — ever | Do not stop for any reason. |
| `CA_COLORED_CURB_BLUE` | Blue | **5** | Disabled person parking only | Valid disabled placard/plates required. |
| `CA_NO_PARKING_15FT_FIRE_HYDRANT` | — | **4** | No parking within 15 ft of fire hydrant | Maintain ≥15 ft clearance from hydrant. |
| `CA_NO_PARKING_SIDEWALK_CROSSWALK` | — | **4** | No parking on sidewalk, in crosswalk, or double-parked | Find legal space. |

### DUI and Impairment (CA)

| Rule ID | Name | Priority | BAC Threshold | Applies To |
|---|---|---|---|---|
| `CA_BAC_LIMIT_21_PLUS` | BAC Limit 0.08% | **0** | ≥0.08% | All drivers 21+ |
| `CA_BAC_LIMIT_UNDER_21` | Zero Tolerance — BAC 0.01% | **0** | ≥0.01% | Drivers under 21 |
| `CA_BAC_LIMIT_COMMERCIAL` | BAC Limit 0.04% (Commercial) | **0** | ≥0.04% | CDL holders and for-hire drivers |
| `CA_OPEN_CONTAINER_LAW` | Open Alcohol Must Be in Trunk | **3** | Any open container | Store in trunk or locked compartment. |

### Child and Occupant Safety (CA)

| Rule ID | Name | Priority | Applies To | Required Action |
|---|---|---|---|---|
| `CA_CHILD_RESTRAINT_REAR_SEAT` | Child Under 8 / Under 4'9" — Rear Seat + Restraint | **1** | Children <8 yrs or <4'9" | Rear seat with appropriate restraint. |
| `CA_CHILD_UNATTENDED_PROHIBITION` | Never Leave Child ≤6 Unattended | **1** | Child 6 years old or younger | Never leave alone (without 12+ yr supervisor). |
| `CA_NO_SMOKING_WITH_MINORS` | No Smoking with Minor in Vehicle | **5** | Minor under 18 in vehicle | Do not smoke in vehicle. |
| `CA_AIRBAG_DISTANCE_10_INCHES` | 10 Inches Between Chest and Airbag | **4** | Driver in any airbag-equipped vehicle | Adjust seat: ≥10 in from steering wheel center. |

### Road Users (CA)

| Rule ID | Name | Priority | When to Apply | Required Action |
|---|---|---|---|---|
| `CA_BICYCLE_3FT_PASSING_CLEARANCE` | 3 Feet Clearance When Passing Cyclist | **2** | Overtaking cyclist | Maintain ≥3 ft lateral clearance. Move lanes if needed. |
| `CA_MOTORCYCLE_LANE_SPLITTING` | Motorcycle Lane Splitting Is Legal | **4** | Motorcycle splitting lanes in slow traffic | Do not block/impede. Check mirrors in congestion. |
| `CA_BICYCLE_NIGHT_EQUIPMENT` | Bicycles Need Lights/Reflectors at Night | **4** | Night driving near cyclists | Scan for lights/reflectors. Reduce speed near bike lanes. |
| `CA_SCHOOL_BUS_DIVIDED_HIGHWAY_EXCEPTION` | School Bus — Opposite Side Divided Highway | **1** | School bus on opposite side of divided road | Oncoming traffic may proceed with caution. |
| `CA_FUNERAL_PROCESSION_YIELD` | Yield to Funeral Procession | **3** | Funeral procession in cross-traffic | Do not cut through. Wait for procession to pass. |
| `CA_EMERGENCY_SCENE_NO_APPROACH` | Do Not Drive to Emergency Scene | **2** | Emergency scene visible ahead | Proceed past. Do not stop to observe. |
| `CA_DIAGONAL_CROSSWALK_ALLOWED` | Diagonal (Scramble) Crossing Allowed | **3** | All-signal-red scramble phase | Remain stopped until all pedestrians clear and signal turns green. |
| `CA_CARGO_FLAGGING_REQUIREMENT` | Flag Cargo Extended Beyond Rear Bumper | **4** | Cargo extending beyond bumper | Red/orange 12-in flag (day); red lights visible 500 ft (night). |

---

## Sign and Marking Name Constants

Use these constants consistently in code and VLA output.

### Signal Names

| Constant | Description |
|---|---|
| `SIGNAL_RED` | Red circular traffic signal |
| `SIGNAL_AMBER` | Amber/yellow circular traffic signal |
| `SIGNAL_GREEN` | Green circular traffic signal |
| `SIGNAL_RED_ARROW` | Red directional arrow signal |
| `SIGNAL_AMBER_ARROW` | Amber directional arrow signal |
| `SIGNAL_GREEN_ARROW` | Green directional arrow signal |
| `SIGNAL_FLASHING_RED` | Flashing red signal (= stop sign) |
| `SIGNAL_FLASHING_AMBER` | Flashing amber signal (= caution) |
| `SIGNAL_DARK` | Non-illuminated signal (= 4-way stop) |
| `SIGNAL_PEDESTRIAN_WALK` | Pedestrian WALK / walking person signal |
| `SIGNAL_PEDESTRIAN_DONT_WALK` | Pedestrian DON'T WALK / raised hand signal |
| `SIGNAL_PEDESTRIAN_COUNTDOWN` | Pedestrian countdown timer signal |
| `SIGNAL_RAG_RED_ARROW` | Singapore RAG system red turn arrow |
| `SIGNAL_RAG_GREEN_ARROW` | Singapore RAG system green turn arrow |
| `SIGNAL_RAILROAD_FLASHING_RED` | Railroad crossing flashing red lights |
| `SIGNAL_SCHOOL_BUS_RED_FLASH` | School bus flashing red lights + STOP arm |

### Road Sign Names

| Constant | Description | Shape/Color |
|---|---|---|
| `SIGN_STOP` | STOP sign | Red octagon, white text |
| `SIGN_YIELD` | YIELD sign | Red/white inverted triangle |
| `SIGN_NO_ENTRY` | No entry | Red circle, white horizontal bar |
| `SIGN_WRONG_WAY` | Wrong way | Red rectangle, white text |
| `SIGN_NO_U_TURN` | No U-turn | Red circle, U-arrow slashed |
| `SIGN_NO_OVERTAKING` | No overtaking | Red circle, two cars slashed |
| `SIGN_NO_PARKING` | No parking | Red circle, P slashed |
| `SIGN_NO_STOPPING` | No stopping | Red circle, line or S slashed |
| `SIGN_NO_WAITING` | No waiting (SG) | Red circle, W |
| `SIGN_NO_TURN_ON_RED` | No turn on red (US/CA) | White rectangle, black text |
| `SIGN_SPEED_LIMIT` | Speed limit sign | White rectangle (US) / Red circle (intl) |
| `SIGN_SPEED_LIMIT_15` | 15 mph speed limit | As above |
| `SIGN_SPEED_LIMIT_25` | 25 mph speed limit | As above |
| `SIGN_TURN_RIGHT_ONLY` | Turn right only (mandatory) | Blue circle, white arrow |
| `SIGN_TURN_LEFT_ONLY` | Turn left only (mandatory) | Blue circle, white arrow |
| `SIGN_AHEAD_ONLY` | Ahead only (mandatory) | Blue circle, white arrow |
| `SIGN_KEEP_LEFT` | Keep left (SG/international) | Blue circle, white arrow |
| `SIGN_KEEP_RIGHT` | Keep right | Blue circle, white arrow |
| `SIGN_ONE_WAY` | One way street | White/black rectangle with arrow |
| `SIGN_PEDESTRIAN_CROSSING` | Pedestrian crossing | Yellow diamond (US) / Triangle (intl) |
| `SIGN_SCHOOL_ZONE` | School zone | Yellow pentagon (US) / Rectangular (SG) |
| `SIGN_CHILDREN_ZONE` | Children zone (SG) | Blue rectangular sign |
| `SIGN_SILVER_ZONE` | Silver zone / elderly area (SG) | Blue rectangular sign |
| `SIGN_WARNING_BEND` | Bend ahead warning | Yellow diamond / red triangle |
| `SIGN_WARNING_ROAD_HUMP` | Road hump ahead | Yellow diamond / red triangle |
| `SIGN_WARNING_SLIPPERY` | Slippery road ahead | Yellow diamond / red triangle |
| `SIGN_WARNING_LEVEL_CROSSING` | Level crossing ahead | Yellow diamond / red triangle |
| `SIGN_WARNING_SCHOOL` | School ahead | Yellow diamond / red triangle |
| `SIGN_WARNING_PEDESTRIAN_CROSSING` | Pedestrian crossing ahead | Yellow diamond / red triangle |
| `SIGN_WARNING_NARROW_ROAD` | Road narrows | Yellow diamond / red triangle |
| `SIGN_WARNING_JUNCTION` | Junction ahead | Yellow diamond / red triangle |
| `SIGN_RAILROAD_CROSSING_ADVANCE` | Railroad crossing advance warning | Yellow X on black |
| `SIGN_RAILROAD_CROSSBUCK` | Railroad crossbuck at crossing | White X with "Railroad Crossing" text |
| `SIGN_ROAD_WORK` | Road work / construction ahead | Orange/red sign |
| `SIGN_FLAGGER_AHEAD` | Flagger ahead in work zone | Orange sign |
| `SIGN_ROUNDABOUT` | Roundabout ahead | Blue circle with white arrow |
| `SIGN_HOV_LANE` | HOV / carpool lane (CA) | White / black sign |
| `SIGN_CLEAN_AIR_VEHICLE` | Clean air vehicle HOV exempt (CA) | Green diamond sign |
| `SIGN_BUS_LANE` | Bus lane (SG) | Red rectangular sign |
| `SIGN_BUS_LANE_FULL_DAY` | Full-day bus lane (SG) | Red rectangular, extended hours |
| `SIGN_BUS_GIVE_WAY` | Give way to bus (SG) | White/red sign |
| `SIGN_BUS_STOP` | Bus stop | Blue/white pole sign |
| `SIGN_SCHOOL_BUS` | School bus stop | Yellow bus sign |
| `SIGN_BIKE_LANE` | Bike lane | White rectangular |
| `SIGN_TOW_AWAY_ZONE` | Tow-away zone (SG/CA) | White/black or white/red |
| `SIGN_WHEEL_CLAMP_ZONE` | Wheel-clamp zone (SG) | White/black sign |
| `SIGN_RED_LIGHT_CAMERA` | Red light camera (SG) | White/black warning sign |
| `SIGN_FIRE_HYDRANT` | Fire hydrant marker | Red/yellow post marker |
| `SIGN_DISABLED_PARKING` | Disabled/accessible parking | Blue rectangle, wheelchair symbol |
| `SIGN_EXPRESSWAY` | Expressway (SG) | Blue shield/route marker |
| `SIGN_MAX_WIDTH` | Maximum width restriction | Circular with width value |
| `SIGN_MAX_HEIGHT` | Maximum height restriction | Circular with height value |
| `SIGN_MAX_WEIGHT` | Maximum weight restriction | Circular with weight value |
| `SIGN_NO_DANGEROUS_GOODS_TUNNEL` | No dangerous goods in tunnel (SG) | Red circle with tanker icon |
| `SIGN_DIAGONAL_CROSSWALK` | Diagonal crossing permitted | Pedestrian signal with diagonal |
| `SIGN_FUNERAL_PROCESSION` | Funeral procession | Vehicle with purple flag |
| `SIGN_EMAS_VARIABLE_SPEED` | EMAS variable speed limit (SG) | Electronic gantry display |
| `SIGN_EMAS_LANE_CLOSED` | EMAS lane closed — X marking (SG) | Red X on overhead gantry |
| `SIGN_EMAS_ACCIDENT_AHEAD` | EMAS accident ahead (SG) | Amber text on LED display |
| `SIGN_EMAS_SLOW_DOWN` | EMAS slow down message (SG) | Amber text on LED display |

### Road Marking Names

| Constant | Description |
|---|---|
| `MARKING_STOP_LINE` | Solid white transverse stop line |
| `MARKING_GIVE_WAY_LINE` | Broken white transverse give-way line |
| `MARKING_CROSSWALK` | Crosswalk (zebra, ladder, or continental pattern) |
| `MARKING_ZEBRA_CROSSING` | Alternating black/white stripe crosswalk (SG) |
| `MARKING_SOLID_YELLOW_CENTER` | Single solid yellow centerline |
| `MARKING_DOUBLE_SOLID_YELLOW` | Double solid yellow centerline (no passing) |
| `MARKING_BROKEN_YELLOW_CENTER` | Broken yellow centerline (passing permitted) |
| `MARKING_SOLID_WHITE` | Solid white lane boundary line |
| `MARKING_DOUBLE_SOLID_WHITE` | Double solid white (HOV boundary CA) |
| `MARKING_BROKEN_WHITE` | Broken white lane divider |
| `MARKING_BIKE_LANE` | Bike lane marking on road surface |
| `MARKING_BUS_LANE_RED` | Red surface marking for bus lane (SG) |
| `MARKING_BUS_PRIORITY_BOX` | Yellow box with BUS PRIORITY text (SG) |
| `MARKING_LANE_ARROW_STRAIGHT` | Straight-ahead arrow in lane |
| `MARKING_LANE_ARROW_LEFT` | Left-turn arrow in lane |
| `MARKING_LANE_ARROW_RIGHT` | Right-turn arrow in lane |
| `MARKING_LANE_ARROW_COMBINED` | Combined direction arrows in lane |
| `MARKING_DIAMOND_HOV` | Diamond symbol in HOV lane (CA) |
| `MARKING_TWO_WAY_LEFT_TURN_LANE` | Two-way left turn lane marking (CA) |
| `MARKING_KEEP_CLEAR` | Keep clear / do not block intersection box |
| `MARKING_YELLOW_BOX` | Yellow box junction — do not block |
| `MARKING_WHITE_ZIGZAG` | White zigzag lines near zebra crossing (SG) |
| `MARKING_YELLOW_ZIGZAG` | Yellow zigzag no-stopping zone (SG) |
| `MARKING_DOUBLE_YELLOW_ZIGZAG` | Double yellow zigzag — absolute no-stop (SG) |
| `MARKING_YELLOW_LINE_SOLID` | Single solid yellow curb line (SG parking) |
| `MARKING_DOUBLE_YELLOW_LINE` | Double yellow curb line — no parking (SG) |
| `MARKING_SLOW_MOVING_VEHICLE_TRIANGLE` | Orange-red triangle reflector on slow vehicle |

### Curb Color Constants (California)

| Constant | Color | Meaning |
|---|---|---|
| `CURB_WHITE` | White | Passenger load/unload only |
| `CURB_GREEN` | Green | Limited time parking |
| `CURB_YELLOW` | Yellow | Freight/passenger loading during business hours |
| `CURB_RED` | Red | No stopping at any time |
| `CURB_BLUE` | Blue | Disabled person parking only |

---

## Quick Reference: Priority Conflict Scenarios

| Scenario | Applicable Rules (in priority order) | Correct Action |
|---|---|---|
| Red light + pedestrian in crosswalk | `STOP_AT_RED_LIGHT` (P1), `PEDESTRIAN_CROSSWALK_ROW` (P1) | Stop for both. Wait for green AND pedestrian to clear. |
| Green light + pedestrian in crosswalk | `PROCEED_GREEN_YIELD_TO_CONFLICTS` (P3), `PEDESTRIAN_CROSSWALK_ROW` (P1) | **Yield to pedestrian** (P1 wins over P3). |
| Emergency vehicle + red light | `EMERGENCY_VEHICLE_PRIORITY` (P0), `STOP_AT_RED_LIGHT` (P1) | **Yield to emergency vehicle** (P0). Pull right and stop. |
| School bus red flash + green light | `SCHOOL_BUS_STOP_RED_LIGHTS` (P1), `PROCEED_GREEN_YIELD_TO_CONFLICTS` (P3) | **Stop for school bus** (P1 wins). |
| Construction zone + posted speed higher | `CONSTRUCTION_ZONE_SPEED_LIMIT` (P2), `OBEY_SPEED_LIMIT_SIGN` (P3) | **Follow construction zone limit** (P2 wins). |
| Roundabout entry + vehicle on right | `ROUNDABOUT_YIELD_TO_INSIDE` (P2), `YIELD_TO_VEHICLE_ON_RIGHT` (P2) | Yield to inside traffic first; yield-to-right is secondary. |
| Bike lane + right turn within 200 ft | `BIKE_LANE_RESTRICTION` (P4), `RIGHT_TURN_FROM_RIGHT_LANE` (P4) | Enter bike lane within 200 ft only. Yield to cyclist. |
| SG bus lane + bus exiting priority box | `SG_PART_DAY_BUS_LANE_RESTRICTION` (P4), `SG_BUS_PRIORITY_BOX_YIELD` (P2) | **Yield to bus** (P2 wins). Do not enter bus lane. |
