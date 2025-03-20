import pandas as pd
import os
import csv
from itertools import product

class VancouverTransitQAGenerator:
    def __init__(self, data_dir):
        """Initialize with directory containing transit CSV files."""
        self.data_dir = data_dir
        self.data = {}
        self.qa_pairs = []
        
    def load_data(self):
        """Load all relevant CSV files from data directory."""
        required_files = [
            'routes.txt', 
            'stops.txt', 
            'trips.txt', 
            'stop_times.txt', 
            'stop_order_exceptions.txt'
        ]
        
        print(f"Looking for transit data files in: {self.data_dir}")
        
        for filename in required_files:
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                print(f"Loading {filename}...")
                self.data[filename.replace('.txt', '')] = pd.read_csv(file_path)
            else:
                print(f"Warning: {filename} not found")
                
        return len(self.data) > 0
    
    def generate_route_questions(self):
        """Generate questions about routes."""
        if 'routes' not in self.data:
            return
            
        routes_df = self.data['routes']
        
        # General questions about all routes
        self.qa_pairs.append({
            'question': "How many bus routes are there in Vancouver?",
            'answer': f"There are {len(routes_df)} bus routes in the Vancouver transit system."
        })
        
        # Questions about specific routes (sample 150)
        for _, route in routes_df.sample(min(150, len(routes_df))).iterrows():
            route_id = route.get('route_id', '')
            route_short_name = route.get('route_short_name', route_id)
            route_long_name = route.get('route_long_name', '')
            
            # Route information questions
            self.qa_pairs.append({
                'question': f"What is route {route_short_name}?",
                'answer': f"Route {route_short_name} is {route_long_name}."
            })
            
            self.qa_pairs.append({
                'question': f"Where does bus {route_short_name} go?",
                'answer': f"Bus {route_short_name} ({route_long_name}) connects various stops in Vancouver. For specific stop information, please ask about this route's stops."
            })
            
        # UBC & major destination specific questions
        ubc_routes = routes_df[routes_df['route_long_name'].str.contains('UBC', na=False)]
        if not ubc_routes.empty:
            ubc_route_names = ", ".join(ubc_routes['route_short_name'].tolist()[:5])
            self.qa_pairs.append({
                'question': "Which buses go to UBC?",
                'answer': f"Buses that go to UBC include routes {ubc_route_names}. These routes connect UBC to various parts of Vancouver."
            })
            
            self.qa_pairs.append({
                'question': "How can I get to UBC by bus?",
                'answer': f"You can reach UBC by taking bus routes {ubc_route_names}. Check the TransLink website or app for the most convenient route based on your starting location."
            })

    def generate_stop_questions(self):
        """Generate questions about stops."""
        if 'stops' not in self.data:
            return
            
        stops_df = self.data['stops']
        
        # General questions about stops
        self.qa_pairs.append({
            'question': "How many bus stops are there in Vancouver?",
            'answer': f"There are {len(stops_df)} bus stops in Vancouver's transit system."
        })
        
        # Sample specific stops (limit to avoid generating too many QA pairs)
        for _, stop in stops_df.sample(min(200, len(stops_df))).iterrows():
            stop_id = stop.get('stop_id', '')
            stop_name = stop.get('stop_name', '')
            stop_lat = stop.get('stop_lat', '')
            stop_lon = stop.get('stop_lon', '')
            
            # Basic stop information
            self.qa_pairs.append({
                'question': f"Where is the {stop_name} bus stop located?",
                'answer': f"The {stop_name} bus stop (ID: {stop_id}) is located at coordinates: {stop_lat}, {stop_lon}."
            })
            
            self.qa_pairs.append({
                'question': f"What is the stop ID for {stop_name}?",
                'answer': f"The stop ID for {stop_name} is {stop_id}."
            })

    def generate_canada_line_questions(self):
        """Generate specific questions about Canada Line based on stop_order_exceptions."""
        if 'stop_order_exceptions' not in self.data:
            return
            
        exceptions_df = self.data['stop_order_exceptions']
        canada_line = exceptions_df[exceptions_df['route_name'] == 'Canada Line']
        
        if not canada_line.empty:
            # Get stations in order
            stations = canada_line.sort_values('stop_do')['stop_name'].tolist()
            stations_str = ", ".join(stations)
            
            self.qa_pairs.append({
                'question': "What are all the stations on the Canada Line?",
                'answer': f"The stations on the Canada Line in order are: {stations_str}."
            })
            
            self.qa_pairs.append({
                'question': "How do I get to the airport by public transit?",
                'answer': "You can take the Canada Line SkyTrain to YVR-Airport Station. The Canada Line connects downtown Vancouver to the airport."
            })
            
            self.qa_pairs.append({
                'question': "How many stations are on the Canada Line?",
                'answer': f"The Canada Line has {len(stations)} stations."
            })

    def generate_trip_questions(self):
        """Generate questions about trips and schedules."""
        if 'trips' not in self.data or 'stop_times' not in self.data:
            return
            
        trips_df = self.data['trips']
        stop_times_df = self.data['stop_times']
        
        # Get some sample trip headsigns
        headsigns = trips_df['trip_headsign'].dropna().unique()[:5]
        
        for headsign in headsigns:
            self.qa_pairs.append({
                'question': f"What is the {headsign} bus route?",
                'answer': f"The {headsign} is a bus route in Vancouver's transit system. It follows a specific path with designated stops."
            })
            
            self.qa_pairs.append({
                'question': f"Where does the {headsign} bus go?",
                'answer': f"The {headsign} bus travels to its specified destination. You can check the TransLink website or app for the exact route map and stops."
            })
        
        # Questions about service frequency
        self.qa_pairs.append({
            'question': "How frequently do buses run in Vancouver?",
            'answer': "Bus frequency in Vancouver varies by route and time of day. Main routes typically run every 5-15 minutes during peak hours and every 15-30 minutes during off-peak times. Night buses run with reduced frequency after regular service hours."
        })

    def generate_student_specific_questions(self):
        """Generate questions specifically helpful for international students."""
        # University area questions
        universities = [
            {"name": "University of British Columbia (UBC)", "routes": ["33", "R4", "49", "14", "4"]},
            {"name": "Simon Fraser University (SFU)", "routes": ["95", "145", "R5"]},
            {"name": "Langara College", "routes": ["15", "49"]},
            {"name": "BCIT", "routes": ["130"]}
        ]
        
        for uni in universities:
            self.qa_pairs.append({
                'question': f"How do I get to {uni['name']} by bus?",
                'answer': f"To get to {uni['name']}, you can take bus routes {', '.join(uni['routes'])}. Always check the TransLink website for the most up-to-date information."
            })
        
        # Student-specific transit questions
        self.qa_pairs.append({
            'question': "Do international students get discounts on Vancouver public transit?",
            'answer': "Yes, international students at recognized post-secondary institutions can get a discounted Compass Card (U-Pass BC) which provides unlimited transit use. This is typically included in your student fees."
        })
        
        self.qa_pairs.append({
            'question': "What is a U-Pass and how do I get one as an international student?",
            'answer': "A U-Pass BC is a discounted transit pass available to eligible students at participating post-secondary institutions in Metro Vancouver. As an international student, you typically receive this as part of your student fees. You'll need to link it to a Compass Card that you can obtain at any SkyTrain station or London Drugs store."
        })
        
        self.qa_pairs.append({
            'question': "How do I pay for the bus in Vancouver?",
            'answer': "In Vancouver, you pay for public transit using a Compass Card, which you can purchase at SkyTrain stations, London Drugs stores, and other locations. As an international student, you're likely eligible for the discounted U-Pass BC program through your institution."
        })
        
        self.qa_pairs.append({
            'question': "Is there a transit app for Vancouver?",
            'answer': "Yes, you can use the TransLink app or Google Maps for Vancouver transit information. The TransLink app provides real-time bus locations, schedules, and trip planning features. Both are very helpful for international students navigating the city."
        })
        
        self.qa_pairs.append({
            'question': "How do I get from Vancouver International Airport to my university?",
            'answer': "From Vancouver International Airport (YVR), take the Canada Line SkyTrain. If you're going to UBC, transfer to bus routes 99 B-Line or 41 at Broadway-City Hall Station. For SFU, transfer to the R5 or 95 B-Line. For other institutions, use the TransLink trip planner for the best route."
        })

    def generate_faq_questions(self):
        """Generate general FAQ questions about Vancouver transit."""
        faqs = [
            {
                "question": "How much does a bus fare cost in Vancouver?",
                "answer": "Vancouver uses a zone-based fare system. A single adult fare is approximately $3.10-$4.45 depending on zones and time of day. As an international student, you'll likely have access to the U-Pass BC program for unlimited travel."
            },
            {
                "question": "What are the operating hours for Vancouver buses?",
                "answer": "Most Vancouver bus routes operate from approximately 5:00 AM to 1:00 AM. Some major routes have NightBus service that runs all night. Schedules vary by route and day of the week."
            },
            {
                "question": "How do I plan a bus trip in Vancouver?",
                "answer": "You can plan your trip using the TransLink website, Google Maps, or the TransLink mobile app. Enter your starting point and destination, and these tools will show you the best routes, including any transfers needed."
            },
            {
                "question": "What's the difference between regular and express buses?",
                "answer": "Regular buses make all stops along their routes. Express buses (marked with an 'X' before the number) make limited stops to provide faster service between major destinations. RapidBus routes (marked with an 'R') offer frequent service with limited stops."
            },
            {
                "question": "How do I get from Vancouver International Airport to downtown?",
                "answer": "The easiest way is to take the Canada Line SkyTrain from the airport to downtown Vancouver. The trip takes about 25 minutes. International students should note that a $5 AddFare applies when departing from the airport stations, but this is waived if you use your U-Pass."
            },
            {
                "question": "Is there a night bus service in Vancouver?",
                "answer": "Yes, Vancouver has a NightBus service identified by routes starting with the letter 'N'. These buses run after regular service hours and provide transportation throughout the night. Major routes include the N8, N15, N20, and N35."
            },
            {
                "question": "How do I find my bus stop?",
                "answer": "Bus stops in Vancouver are marked with bus stop signs that display the route numbers. Many stops have shelters with route maps and schedules. You can also use the TransLink app or Google Maps to locate the nearest bus stop to your location."
            },
            {
                "question": "What should I do if I lose something on a Vancouver bus?",
                "answer": "Contact TransLink's Lost Property Office at 604-953-3334 or visit the Lost Property Office at Stadium-Chinatown SkyTrain station. Be prepared to describe the item and provide details about the route, time, and date you lost it."
            },
            {
                "question": "How do I get to Stanley Park by public transit?",
                "answer": "You can take bus routes 19 or 23 to Stanley Park. Both routes stop near the park entrance. From downtown, you can also walk or take the #19 bus along Pender Street."
            },
            {
                "question": "How do I transfer between buses in Vancouver?",
                "answer": "When using a Compass Card or U-Pass, transfers are automatic and valid for 90 minutes from the time you first tap your card. During this time, you can transfer between buses, SkyTrain, and SeaBus without paying again."
            }
        ]
        
        self.qa_pairs.extend(faqs)
        
    def generate_weather_related_questions(self):
        """Generate questions about transit during Vancouver weather conditions."""
        weather_qs = [
            {
                "question": "Do buses in Vancouver run when it snows?",
                "answer": "Yes, buses in Vancouver typically continue to run during snowfall, though there may be delays and route changes. TransLink implements snow plans that may include detours around steep hills. Check the TransLink website or app for service alerts during snowy conditions."
            },
            {
                "question": "How reliable is public transit in Vancouver during rainy season?",
                "answer": "Vancouver's public transit system generally operates reliably during the rainy season (October to March). Rain rarely causes significant disruptions to bus or SkyTrain service. However, heavier rainfall may lead to minor delays due to increased traffic and reduced visibility."
            }
        ]
        
        self.qa_pairs.extend(weather_qs)

    def generate_all_qa_pairs(self):
        """Generate all types of QA pairs."""
        self.generate_route_questions()
        self.generate_stop_questions()
        self.generate_canada_line_questions()
        self.generate_trip_questions()
        self.generate_student_specific_questions()
        self.generate_faq_questions()
        self.generate_weather_related_questions()
        
        print(f"Generated {len(self.qa_pairs)} Q&A pairs.")
        return self.qa_pairs
    
    def save_qa_pairs(self, output_file="vancouver_transit_qa_pairs.csv"):
        """Save the generated QA pairs to a CSV file."""
        output_path = os.path.join(self.data_dir, output_file)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])
            
            for qa in self.qa_pairs:
                writer.writerow([qa['question'], qa['answer']])
        
        print(f"Saved {len(self.qa_pairs)} Q&A pairs to {output_path}")
        return output_path

def main():
    # Directory containing transit data files
    data_dir = input("Enter the directory path containing Vancouver transit CSV files: ")
    
    # Create QA generator
    generator = VancouverTransitQAGenerator(data_dir)
    
    # Load transit data
    if not generator.load_data():
        print("No transit data files found. Please check the directory path.")
        return
    
    # Generate QA pairs
    generator.generate_all_qa_pairs()
    
    # Save to CSV
    output_file = generator.save_qa_pairs()
    print(f"Q&A pairs saved to {output_file}")

if __name__ == "__main__":
    main()
