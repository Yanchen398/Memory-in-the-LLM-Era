"""
Tool definitions for OpenAI Tools API (function calling)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "FlightSearch",
            "description": "Search for flights between two cities on a specific date. Returns available flights with price, departure time, arrival time, and duration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin city name (e.g., 'New York', 'Los Angeles')"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination city name (e.g., 'Chicago', 'Miami')"
                    },
                    "date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format (e.g., '2022-03-15')"
                    }
                },
                "required": ["origin", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "RestaurantSearch",
            "description": "Search for restaurants in a specific city. Returns restaurant names, cuisines, average cost, and ratings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name to search restaurants in (e.g., 'Rockford', 'Pensacola')"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "AccommodationSearch",
            "description": "Search for accommodations (hotels, apartments) in a specific city. Returns accommodation names, room types, prices, and house rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name to search accommodations in (e.g., 'Rockford', 'Pensacola')"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "AttractionSearch",
            "description": "Search for tourist attractions in a specific city. Returns attraction names, addresses, and contact information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name to search attractions in (e.g., 'Rockford', 'Pensacola')"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "DistanceMatrix",
            "description": "Get driving distance and duration between two cities. Useful for planning self-driving trips.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Origin city name"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination city name"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["driving", "taxi"],
                        "description": "Travel mode: 'driving' for self-driving, 'taxi' for taxi"
                    }
                },
                "required": ["origin", "destination", "mode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "CitySearch",
            "description": "Search for cities in a specific US state. Returns a list of city names in that state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "US state name (e.g., 'California', 'Texas', 'Florida')"
                    }
                },
                "required": ["state"]
            }
        }
    }
]

# Tool name to display name mapping
TOOL_DISPLAY_NAMES = {
    "FlightSearch": "Flight Search",
    "RestaurantSearch": "Restaurant Search",
    "AccommodationSearch": "Accommodation Search",
    "AttractionSearch": "Attraction Search",
    "DistanceMatrix": "Distance Matrix",
    "CitySearch": "City Search"
}

