"""
Tool Executor - Executes tool calls by invoking the corresponding APIs
"""

import os

from .tools.flights import Flights
from .tools.restaurants import Restaurants
from .tools.accommodations import Accommodations
from .tools.attractions import Attractions
from .tools.distance_matrix import GoogleDistanceMatrix
from .tools.cities import Cities

ENV_DIR = os.path.dirname(os.path.abspath(__file__))


class ToolExecutor:
    """
    Executes tool calls by routing to the appropriate API.
    All tools share the same database instances.
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize all tool APIs.
        """
        if db_path is None:
            db_path = os.path.join(ENV_DIR, "database")
        self.db_path = db_path
        
        self.flights = Flights(path=f"{self.db_path}/flights/clean_Flights_2022.csv")
        self.restaurants = Restaurants(path=f"{self.db_path}/restaurants/clean_restaurant_2022.csv")
        self.accommodations = Accommodations(path=f"{self.db_path}/accommodations/clean_accommodations_2022.csv")
        self.attractions = Attractions(path=f"{self.db_path}/attractions/attractions.csv")
        self.distance_matrix = GoogleDistanceMatrix(path=f"{self.db_path}/googleDistanceMatrix/distance.csv")
        self.cities = Cities(path=f"{self.db_path}/background/citySet_with_states.txt")
        
        print("ToolExecutor initialized with all APIs.")
    
    def execute(self, tool_name: str, args: dict) -> str:
        """
        Execute a tool call and return the result as string.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool (parsed from JSON)
            
        Returns:
            String representation of the tool result
        """
        try:
            if tool_name == "FlightSearch":
                result = self.flights.run(
                    origin=args['origin'],
                    destination=args['destination'],
                    departure_date=args['date']
                )
            
            elif tool_name == "RestaurantSearch":
                result = self.restaurants.run(city=args['city'])
            
            elif tool_name == "AccommodationSearch":
                result = self.accommodations.run(city=args['city'])
            
            elif tool_name == "AttractionSearch":
                result = self.attractions.run(city=args['city'])
            
            elif tool_name == "DistanceMatrix":
                if args.get('mode') == 'taxi':
                    result = self.distance_matrix.run_for_evaluation(
                        origin=args['origin'],
                        destination=args['destination'],
                        mode='taxi'
                    )
                else:
                    result = self.distance_matrix.run_for_evaluation(
                        origin=args['origin'],
                        destination=args['destination'],
                        mode='driving'
                    )
            
            elif tool_name == "CitySearch":
                result = self.cities.run(state=args['state'])
            
            else:
                return f"Error: Unknown tool '{tool_name}'"
            
            return self._format_result(result, tool_name)
            
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def _format_result(self, result, tool_name: str) -> str:
        """
        Format the result as a readable string.
        """
        if isinstance(result, str):
            return result
        
        if hasattr(result, 'to_string'):
            return result.to_string(index=False)
        
        if isinstance(result, dict):
            lines = []
            for key, value in result.items():
                lines.append(f"{key}: {value}")
            return "\n".join(lines)
        
        if isinstance(result, list):
            return ", ".join(str(item) for item in result)
        
        return str(result)
