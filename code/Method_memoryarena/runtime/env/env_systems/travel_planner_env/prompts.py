"""
Prompt templates for travel planning agents
"""

# ============================================================
# Base System Prompt (for direct planning with reference info)
# ============================================================
SYSTEM_PROMPT = """You are a proficient planner for group travel. Based on the provided information and queries from travelers, please give me a detailed travel plan.

**Important**: This is a multi-turn conversation. The first traveler's plan is already finalized and provided to you. You need to generate plans for ALL travelers EXCEPT the first one (whose plan is already fixed).

You may adjust previous travelers' plans if needed to accommodate new constraints, but the first traveler's plan is FIXED and cannot be changed.

For example:
- Traveler A's plan is already finalized (given to you, do NOT output or modify it)
- Turn 1 (Traveler B joins): Output B's plan
- Turn 2 (Traveler C joins): Output B's AND C's plans (you may adjust B's plan if needed)
- Turn 3 (Traveler D joins): Output B's, C's, AND D's plans (you may adjust B's and C's plans if needed)

Include specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. All information must be derived from the provided data. The symbol '-' indicates that information is unnecessary. When traveling to two cities in one day, note it in 'Current City' (e.g., from A to B).

Do NOT include explanatory details like prices, ratings, or cuisines in the output.

**STRICT FORMAT RULES - Follow EXACTLY:**
- Use "Day 1:", "Day 2:", "Day 3:" etc. Do NOT add dates in parentheses like "(2022-03-12)".
- Do NOT add any extra text, comments, or explanations.
- Follow the example format precisely, no variations.

***** Example (Alice's plan is already finalized, Bob joins) *****

=== Bob's Plan ===
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:38, Arrival Time: 07:46
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte;Romare Bearden Park, Charlotte
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -

***** Example Ends *****"""


# ============================================================
# Agent System Prompt (for tool-using agent)
# ============================================================
AGENT_SYSTEM_PROMPT = """

You are a travel planner assistant. Your task is to create travel plans using the available tools.

You have access to these tools:
- FlightSearch: Search for flights between cities on a specific date
- RestaurantSearch: Search for restaurants in a city
- AccommodationSearch: Search for accommodations in a city
- AttractionSearch: Search for tourist attractions in a city
- DistanceMatrix: Get driving distance and time between cities
- CitySearch: Search for cities in a specific US state

**WORKFLOW:**
1. First, use the tools to search for available flights, restaurants, accommodations, and attractions
2. Then, output the final plan in the EXACT format specified below

**CRITICAL: OUTPUT FORMAT (MUST FOLLOW EXACTLY)**

Your final output MUST start with "=== {Name}'s Plan ===" and follow this EXACT structure:

=== {Name}'s Plan ===
Day 1:
Current City: from {origin} to {destination}
Transportation: Flight Number: {flight_number}, from {origin} to {destination}, Departure Time: {dep_time}, Arrival Time: {arr_time}
Breakfast: {restaurant_name}, {city}
Attraction: {attraction1}, {city};{attraction2}, {city}
Lunch: {restaurant_name}, {city}
Dinner: {restaurant_name}, {city}
Accommodation: {accommodation_name}, {city}

Day 2:
Current City: {city}
Transportation: -
Breakfast: {restaurant_name}, {city}
Attraction: {attraction1}, {city};{attraction2}, {city}
Lunch: {restaurant_name}, {city}
Dinner: {restaurant_name}, {city}
Accommodation: {accommodation_name}, {city}

Day 3:
Current City: from {destination} to {origin}
Transportation: Flight Number: {flight_number}, from {destination} to {origin}, Departure Time: {dep_time}, Arrival Time: {arr_time}
Breakfast: {restaurant_name}, {city}
Attraction: {attraction1}, {city}
Lunch: {restaurant_name}, {city}
Dinner: {restaurant_name}, {city}
Accommodation: -

**RULES:**
- Use "-" for fields that don't apply (e.g., no transportation on days staying in same city, no accommodation on last day)
- All restaurant/hotel/attraction names MUST exactly match the search results
- Do NOT include prices, ratings, or any extra explanations
- Do NOT add any text before or after the plan
- Multiple attractions on same day: separate with semicolon (;)
"""


# ============================================================
# User Prompt Templates
# ============================================================

# When reference info is provided (direct planning mode)
USER_PROMPT_TEMPLATE = """Given information: {text}
Query: {query}
Travel Plan:"""

# When agent needs to use tools (agent mode)
AGENT_USER_PROMPT_TEMPLATE = """Create a travel plan for {name}.

Query: {query}

Instructions:
1. Use the tools to search for flights, restaurants, accommodations, and attractions
2. After gathering information, output the plan in the EXACT format starting with "=== {name}'s Plan ==="
3. Do NOT include any explanations, summaries, or extra text - ONLY output the formatted plan"""


# ============================================================
# History and Context Templates
# ============================================================

# For multi-round: include all queries, previous plans, and judgement
HISTORY_TEMPLATE = """=== All Travelers' Queries ===
{all_queries}

=== Previous Plan ===
{previous_plan}

=== Judgement ===
{judgement}

=== New Traveler ===
"""

# For first round: explain base person's plan is already fixed
BASE_PERSON_TEMPLATE = """=== {base_name}'s Request (Already Planned) ===
{base_name} has already made their travel request and their plan has been finalized.

{base_name}'s Query: {base_query}

=== {base_name}'s Confirmed Plan ===
{base_plan}

Now other travelers are joining the trip. Generate plans for all travelers except {base_name} (whose plan is already finalized above).

=== New Traveler Joining ===
"""
