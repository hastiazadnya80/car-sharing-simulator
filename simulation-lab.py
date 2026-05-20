import numpy as np
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from collections import defaultdict

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# LAB 2: MODEL DESIGN - Core Classes and Data Structures
# ============================================================================

class EventType(Enum):
    """Types of events in the simulation"""
    USER_ARRIVAL = 1
    TRIP_START = 2
    TRIP_END = 3
    RELOCATION_START = 4
    RELOCATION_END = 5

@dataclass(order=True)
class Event:
    """Event in the Future Event Set (FES)"""
    time: float
    event_type: EventType = field(compare=False)
    user_id: int = field(default=None, compare=False)
    car_id: int = field(default=None, compare=False)
    location: Tuple[float, float] = field(default=None, compare=False)
    destination: Tuple[float, float] = field(default=None, compare=False)

class Car:
    """Represents a car in the fleet"""
    def __init__(self, car_id: int, location: Tuple[float, float]):
        self.id = car_id
        self.location = location
        self.is_available = True
        self.current_user = None
        self.total_trips = 0
        self.total_distance = 0
        self.last_trip_end_time = 0

class UserRequest:
    """Represents a user request"""
    def __init__(self, user_id: int, arrival_time: float,
                 origin: Tuple[float, float], destination: Tuple[float, float]):
        self.id = user_id
        self.arrival_time = arrival_time
        self.origin = origin
        self.destination = destination
        self.assigned_car = None
        self.service_start_time = None
        self.service_end_time = None

class Zone:
    """Represents a spatial zone in the service area"""
    def __init__(self, zone_id: int, center: Tuple[float, float], radius: float):
        self.id = zone_id
        self.center = center
        self.radius = radius
        self.demand_count = 0
        self.supply_count = 0

# ============================================================================
# LAB 3: IMPLEMENTATION - Main Simulator
# ============================================================================

class CarSharingSimulator:
    """
    Discrete Event Simulation of a Car-Sharing System

    Key Parameters (Lab 2):
    - fleet_size: Number of cars in the system
    - area_size: Size of the service area (square)
    - user_arrival_rate: Average users per hour
    - avg_trip_duration: Average trip time in minutes
    - relocation_enabled: Whether to use car relocation
    - relocation_threshold: Trigger relocation when availability drops below this
    - relocation_check_interval: How often to check for relocation needs (minutes)
    """

    def __init__(self, fleet_size: int = 100, area_size: float = 10.0,
                 user_arrival_rate: float = 50.0, avg_trip_duration: float = 20.0,
                 avg_speed: float = 30.0, relocation_enabled: bool = False,
                 relocation_threshold: float = 0.2, relocation_check_interval: float = 30.0,
                 num_zones: int = 4, random_seed: int = None):

        # System parameters
        self.fleet_size = fleet_size
        self.area_size = area_size
        self.user_arrival_rate = user_arrival_rate  # users per hour
        self.avg_trip_duration = avg_trip_duration  # minutes
        self.avg_speed = avg_speed  # km/h
        self.relocation_enabled = relocation_enabled
        self.relocation_threshold = relocation_threshold
        self.relocation_check_interval = relocation_check_interval
        self.num_zones = num_zones

        # ENHANCEMENT 1: Random seed for reproducibility
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Simulation state
        self.current_time = 0.0
        self.fes = []  # Future Event Set (priority queue)
        self.cars = []
        self.pending_requests = []
        self.completed_requests = []
        self.zones = []

        # Statistics (Performance Metrics from Lab 2)
        self.total_requests = 0
        self.successful_assignments = 0
        self.total_waiting_time = 0.0
        self.waiting_times = []
        self.availability_samples = []
        self.utilization_samples = []
        self.time_series_data = []  # For transient detection
        self.relocation_events = []  # Track relocation events

        # ENHANCEMENT 2: Zone-based demand tracking for relocation
        self._initialize_zones()
        self.zone_demand_history = defaultdict(list)
        self.zone_supply_history = defaultdict(list)

        # Initialize cars with random locations
        for i in range(fleet_size):
            loc = (np.random.uniform(0, area_size),
                   np.random.uniform(0, area_size))
            self.cars.append(Car(i, loc))

    def _initialize_zones(self):
        """Initialize spatial zones for demand/supply tracking"""
        # Create grid of zones
        zones_per_side = int(np.sqrt(self.num_zones))
        zone_width = self.area_size / zones_per_side

        zone_id = 0
        for i in range(zones_per_side):
            for j in range(zones_per_side):
                center_x = (i + 0.5) * zone_width
                center_y = (j + 0.5) * zone_width
                self.zones.append(Zone(zone_id, (center_x, center_y), zone_width / 2))
                zone_id += 1

    def get_zone(self, location: Tuple[float, float]) -> Zone:
        """Find which zone a location belongs to"""
        return min(self.zones, key=lambda z: self.distance(z.center, location))

    def distance(self, loc1: Tuple[float, float],
                 loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

    def find_nearest_car(self, location: Tuple[float, float]) -> Optional[Car]:
        """Find nearest available car to a location"""
        available_cars = [c for c in self.cars if c.is_available]
        if not available_cars:
            return None
        return min(available_cars, key=lambda c: self.distance(c.location, location))

    def schedule_event(self, event: Event):
        """Add event to the Future Event Set"""
        heapq.heappush(self.fes, event)

    def generate_user_arrival(self):
        """Generate next user arrival using exponential distribution"""
        inter_arrival = np.random.exponential(60.0 / self.user_arrival_rate)
        arrival_time = self.current_time + inter_arrival

        # Generate random origin and destination
        origin = (np.random.uniform(0, self.area_size),
                  np.random.uniform(0, self.area_size))
        destination = (np.random.uniform(0, self.area_size),
                       np.random.uniform(0, self.area_size))

        event = Event(
            time=arrival_time,
            event_type=EventType.USER_ARRIVAL,
            user_id=self.total_requests,
            location=origin,
            destination=destination
        )
        self.schedule_event(event)

    def handle_user_arrival(self, event: Event):
        """Process user arrival event"""
        self.total_requests += 1

        # Update zone demand
        zone = self.get_zone(event.location)
        zone.demand_count += 1

        # Create user request
        request = UserRequest(event.user_id, event.time,
                             event.location, event.destination)

        # Try to assign nearest car
        car = self.find_nearest_car(event.location)

        if car:
            # Successful assignment
            self.successful_assignments += 1
            car.is_available = False
            car.current_user = request.id
            request.assigned_car = car.id
            request.service_start_time = event.time

            # Calculate waiting time (time to reach user)
            dist_to_user = self.distance(car.location, event.location)
            waiting_time = (dist_to_user / self.avg_speed) * 60  # minutes
            self.total_waiting_time += waiting_time
            self.waiting_times.append(waiting_time)

            # Move car to user location
            car.location = event.location

            # Schedule trip end
            trip_distance = self.distance(event.location, event.destination)
            trip_time = (trip_distance / self.avg_speed) * 60  # minutes

            end_event = Event(
                time=event.time + trip_time,
                event_type=EventType.TRIP_END,
                user_id=request.id,
                car_id=car.id,
                destination=event.destination
            )
            self.schedule_event(end_event)
        else:
            # No car available - request denied
            self.pending_requests.append(request)

        # Generate next arrival
        self.generate_user_arrival()

    def handle_trip_end(self, event: Event):
        """Process trip end event"""
        car = self.cars[event.car_id]
        car.location = event.destination
        car.is_available = True
        car.total_trips += 1
        car.current_user = None
        car.last_trip_end_time = event.time

        # Update zone supply
        zone = self.get_zone(event.destination)
        zone.supply_count += 1

    # ENHANCEMENT 2: Actual Relocation Logic Implementation
    def check_and_relocate(self):
        """Check zones and relocate cars if needed"""
        if not self.relocation_enabled:
            return

        # Calculate demand and supply imbalance per zone
        zone_imbalance = []
        for zone in self.zones:
            # Calculate recent demand/supply ratio
            demand = zone.demand_count
            supply = sum(1 for car in self.cars
                        if car.is_available and self.get_zone(car.location).id == zone.id)

            # Imbalance = demand - supply (positive means shortage)
            imbalance = demand - supply
            zone_imbalance.append((zone, imbalance, supply))

        # Sort zones: highest shortage first, then highest surplus
        shortage_zones = sorted([z for z in zone_imbalance if z[1] > 0],
                               key=lambda x: x[1], reverse=True)
        surplus_zones = sorted([z for z in zone_imbalance if z[1] < 0 and z[2] > 0],
                              key=lambda x: x[1])

        # Relocate cars from surplus to shortage zones
        relocations_done = 0
        for shortage_zone, shortage, _ in shortage_zones:
            if not surplus_zones or relocations_done >= 3:  # Limit relocations per check
                break

            for surplus_zone, surplus, available in surplus_zones:
                if available <= 1:  # Keep at least 1 car in each zone
                    continue

                # Find a car in the surplus zone
                available_cars = [c for c in self.cars
                                 if c.is_available and
                                 self.get_zone(c.location).id == surplus_zone.id]

                if available_cars:
                    # Pick the car that has been idle longest
                    car_to_relocate = max(available_cars,
                                         key=lambda c: self.current_time - c.last_trip_end_time)

                    # Schedule relocation
                    self._schedule_relocation(car_to_relocate, shortage_zone.center)
                    relocations_done += 1

                    # Update surplus zone supply count
                    surplus_zone.supply_count -= 1

                    break

        # Reset zone counters for next period
        for zone in self.zones:
            zone.demand_count = 0
            zone.supply_count = 0

    def _schedule_relocation(self, car: Car, destination: Tuple[float, float]):
        """Schedule a car relocation event"""
        car.is_available = False  # Car becomes unavailable during relocation

        relocation_distance = self.distance(car.location, destination)
        relocation_time = (relocation_distance / self.avg_speed) * 60  # minutes

        relocation_event = Event(
            time=self.current_time + relocation_time,
            event_type=EventType.RELOCATION_END,
            car_id=car.id,
            destination=destination
        )
        self.schedule_event(relocation_event)

        # Track relocation for analysis
        self.relocation_events.append({
            'time': self.current_time,
            'car_id': car.id,
            'from': car.location,
            'to': destination,
            'distance': relocation_distance
        })

    def handle_relocation_end(self, event: Event):
        """Process relocation end event"""
        car = self.cars[event.car_id]
        car.location = event.destination
        car.is_available = True
        car.last_trip_end_time = event.time

    def collect_statistics(self):
        """Collect performance metrics at current time"""
        available = sum(c.is_available for c in self.cars)
        availability_rate = available / self.fleet_size
        utilization_rate = 1.0 - availability_rate

        self.availability_samples.append(availability_rate)
        self.utilization_samples.append(utilization_rate)

        # Track zone-level supply/demand
        for zone in self.zones:
            supply = sum(1 for car in self.cars
                        if car.is_available and self.get_zone(car.location).id == zone.id)
            self.zone_supply_history[zone.id].append(supply)

        # For transient detection - record key metric over time
        avg_wait = np.mean(self.waiting_times[-100:]) if len(self.waiting_times) >= 100 else 0
        self.time_series_data.append((self.current_time, avg_wait, utilization_rate))

    def run(self, simulation_time: float, warmup_time: float = 0.0):
        """
        Run simulation for specified time (in minutes)
        warmup_time: time to ignore for statistics (transient period)
        """
        # Initialize with first arrival
        self.generate_user_arrival()

        # Schedule periodic relocation checks
        if self.relocation_enabled:
            next_check = self.relocation_check_interval
            while next_check < simulation_time:
                event = Event(
                    time=next_check,
                    event_type=EventType.RELOCATION_START,
                )
                self.schedule_event(event)
                next_check += self.relocation_check_interval

        # Main simulation loop
        while self.fes and self.current_time < simulation_time:
            # Get next event
            event = heapq.heappop(self.fes)
            self.current_time = event.time

            if self.current_time > simulation_time:
                break

            # Process event
            if event.event_type == EventType.USER_ARRIVAL:
                self.handle_user_arrival(event)
            elif event.event_type == EventType.TRIP_END:
                self.handle_trip_end(event)
            elif event.event_type == EventType.RELOCATION_START:
                self.check_and_relocate()
            elif event.event_type == EventType.RELOCATION_END:
                self.handle_relocation_end(event)

            # Collect statistics every 5 minutes (after warmup)
            if self.current_time > warmup_time and int(self.current_time) % 5 == 0:
                self.collect_statistics()

    def get_performance_metrics(self) -> dict:
        """Calculate and return performance metrics (Lab 2 KPIs)"""
        if self.successful_assignments == 0:
            return {}

        return {
            'total_requests': self.total_requests,
            'successful_assignments': self.successful_assignments,
            'availability_rate': self.successful_assignments / self.total_requests,
            'avg_waiting_time': self.total_waiting_time / self.successful_assignments,
            'avg_utilization': np.mean(self.utilization_samples) if self.utilization_samples else 0,
            'total_trips': sum(c.total_trips for c in self.cars),
            'total_relocations': len(self.relocation_events)
        }

    # ENHANCEMENT 4: Add transient detection overlay in evolution plots
    def plot_system_evolution(self, transient_end: Optional[int] = None):
        """Plot key metrics evolution over time with optional transient overlay"""
        if not self.time_series_data:
            print("No time series data available")
            return

        times, waiting_times, utilizations = zip(*self.time_series_data)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Car-Sharing System Evolution Over Time', fontsize=16, fontweight='bold')

        # Plot 1: Waiting Time Evolution with transient overlay
        axes[0, 0].plot(times, waiting_times, color='#2E86AB', linewidth=2, alpha=0.7)
        if transient_end is not None and transient_end < len(times):
            transient_time = times[transient_end]
            axes[0, 0].axvline(x=transient_time, color='red', linestyle='--',
                              linewidth=2, label=f'Transient End ({transient_time:.0f} min)')
            axes[0, 0].axvspan(0, transient_time, alpha=0.2, color='red', label='Transient Phase')
            axes[0, 0].axvspan(transient_time, times[-1], alpha=0.2, color='green', label='Steady State')
        axes[0, 0].set_xlabel('Time (minutes)', fontsize=11)
        axes[0, 0].set_ylabel('Avg Waiting Time (min)', fontsize=11)
        axes[0, 0].set_title('Waiting Time Evolution', fontweight='bold')
        axes[0, 0].legend(loc='best', fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Utilization Rate Evolution with transient overlay
        axes[0, 1].plot(times, utilizations, color='#A23B72', linewidth=2, alpha=0.7)
        if transient_end is not None and transient_end < len(times):
            transient_time = times[transient_end]
            axes[0, 1].axvline(x=transient_time, color='red', linestyle='--',
                              linewidth=2, label=f'Transient End')
            axes[0, 1].axvspan(0, transient_time, alpha=0.2, color='red')
            axes[0, 1].axvspan(transient_time, times[-1], alpha=0.2, color='green')
        axes[0, 1].set_xlabel('Time (minutes)', fontsize=11)
        axes[0, 1].set_ylabel('Utilization Rate', fontsize=11)
        axes[0, 1].set_title('Fleet Utilization Over Time', fontweight='bold')
        axes[0, 1].legend(loc='best', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        # Plot 3: Waiting Time Distribution
        if len(self.waiting_times) > 0:
            # Separate transient and steady-state data
            if transient_end is not None and transient_end < len(self.waiting_times):
                axes[1, 0].hist(self.waiting_times[:transient_end], bins=30,
                               color='red', alpha=0.5, label='Transient', edgecolor='black')
                axes[1, 0].hist(self.waiting_times[transient_end:], bins=30,
                               color='green', alpha=0.5, label='Steady State', edgecolor='black')
            else:
                axes[1, 0].hist(self.waiting_times, bins=50, color='#F18F01',
                               alpha=0.7, edgecolor='black')

            axes[1, 0].axvline(np.mean(self.waiting_times), color='blue', linestyle='--',
                              linewidth=2, label=f'Mean: {np.mean(self.waiting_times):.2f} min')
            axes[1, 0].set_xlabel('Waiting Time (minutes)', fontsize=11)
            axes[1, 0].set_ylabel('Frequency', fontsize=11)
            axes[1, 0].set_title('Waiting Time Distribution', fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Car Utilization Distribution
        trip_counts = [car.total_trips for car in self.cars]
        axes[1, 1].hist(trip_counts, bins=30, color='#6A994E', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(trip_counts), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(trip_counts):.1f} trips')
        axes[1, 1].set_xlabel('Number of Trips', fontsize=11)
        axes[1, 1].set_ylabel('Number of Cars', fontsize=11)
        axes[1, 1].set_title('Car Usage Distribution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_car_locations(self):
        """Visualize car locations in the service area"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Car Distribution in Service Area', fontsize=16, fontweight='bold')

        # Current car locations
        available_cars = [c for c in self.cars if c.is_available]
        busy_cars = [c for c in self.cars if not c.is_available]

        # Draw zones
        for zone in self.zones:
            circle = plt.Circle(zone.center, zone.radius, color='gray',
                               fill=False, linestyle='--', alpha=0.3)
            axes[0].add_patch(circle)
            axes[0].text(zone.center[0], zone.center[1], f'Z{zone.id}',
                        ha='center', va='center', fontsize=8, alpha=0.5)

        if available_cars:
            avail_x, avail_y = zip(*[c.location for c in available_cars])
            axes[0].scatter(avail_x, avail_y, c='green', s=100, alpha=0.6,
                           label=f'Available ({len(available_cars)})', marker='o')

        if busy_cars:
            busy_x, busy_y = zip(*[c.location for c in busy_cars])
            axes[0].scatter(busy_x, busy_y, c='red', s=100, alpha=0.6,
                           label=f'In Use ({len(busy_cars)})', marker='s')

        axes[0].set_xlim([0, self.area_size])
        axes[0].set_ylim([0, self.area_size])
        axes[0].set_xlabel('X Coordinate (km)', fontsize=11)
        axes[0].set_ylabel('Y Coordinate (km)', fontsize=11)
        axes[0].set_title('Current Car Locations', fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')

        # Heatmap of car density
        x_coords = [c.location[0] for c in self.cars]
        y_coords = [c.location[1] for c in self.cars]

        heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=20,
                                                  range=[[0, self.area_size], [0, self.area_size]])

        im = axes[1].imshow(heatmap.T, origin='lower', cmap='YlOrRd',
                           extent=[0, self.area_size, 0, self.area_size], alpha=0.8)

        # Draw zone boundaries on heatmap
        for zone in self.zones:
            circle = plt.Circle(zone.center, zone.radius, color='black',
                               fill=False, linestyle='--', alpha=0.3)
            axes[1].add_patch(circle)

        axes[1].set_xlabel('X Coordinate (km)', fontsize=11)
        axes[1].set_ylabel('Y Coordinate (km)', fontsize=11)
        axes[1].set_title('Car Density Heatmap', fontweight='bold')
        plt.colorbar(im, ax=axes[1], label='Number of Cars')
        axes[1].set_aspect('equal')

        plt.tight_layout()
        return fig

    def plot_relocation_analysis(self):
        """Visualize relocation activity and effectiveness"""
        if not self.relocation_events:
            print("No relocation events to plot (relocation disabled or no relocations occurred)")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Relocation Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Relocation events over time
        times = [r['time'] for r in self.relocation_events]
        axes[0, 0].scatter(times, range(len(times)), alpha=0.6, s=50, color='#2E86AB')
        axes[0, 0].set_xlabel('Time (minutes)', fontsize=11)
        axes[0, 0].set_ylabel('Relocation Event Number', fontsize=11)
        axes[0, 0].set_title(f'Relocation Events Over Time (Total: {len(times)})', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Relocation distances
        distances = [r['distance'] for r in self.relocation_events]
        axes[0, 1].hist(distances, bins=20, color='#F18F01', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(distances), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(distances):.2f} km')
        axes[0, 1].set_xlabel('Relocation Distance (km)', fontsize=11)
        axes[0, 1].set_ylabel('Frequency', fontsize=11)
        axes[0, 1].set_title('Distribution of Relocation Distances', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Relocation map (from/to locations)
        for reloc in self.relocation_events[:50]:  # Limit to first 50 for clarity
            from_loc = reloc['from']
            to_loc = reloc['to']
            axes[1, 0].arrow(from_loc[0], from_loc[1],
                           to_loc[0] - from_loc[0], to_loc[1] - from_loc[1],
                           head_width=0.2, head_length=0.3, fc='blue',
                           ec='blue', alpha=0.3, length_includes_head=True)

        # Draw zones
        for zone in self.zones:
            circle = plt.Circle(zone.center, zone.radius, color='gray',
                               fill=False, linestyle='--', alpha=0.5)
            axes[1, 0].add_patch(circle)

        axes[1, 0].set_xlim([0, self.area_size])
        axes[1, 0].set_ylim([0, self.area_size])
        axes[1, 0].set_xlabel('X Coordinate (km)', fontsize=11)
        axes[1, 0].set_ylabel('Y Coordinate (km)', fontsize=11)
        axes[1, 0].set_title('Relocation Patterns (First 50)', fontweight='bold')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Cars relocated per check interval
        time_bins = np.arange(0, max(times) + self.relocation_check_interval,
                             self.relocation_check_interval)
        relocations_per_interval, _ = np.histogram(times, bins=time_bins)

        axes[1, 1].bar(time_bins[:-1], relocations_per_interval,
                      width=self.relocation_check_interval*0.8,
                      color='#6A994E', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Time (minutes)', fontsize=11)
        axes[1, 1].set_ylabel('Number of Relocations', fontsize=11)
        axes[1, 1].set_title(f'Relocations per {self.relocation_check_interval:.0f}-min Interval',
                           fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

# ============================================================================
# LAB 4: TRANSIENT DETECTION & CONFIDENCE INTERVALS
# ============================================================================

class TransientDetector:
    """
    Implements automated transient phase detection using Welch's method
    """

    @staticmethod
    def welch_method(data: List[float], window_size: int = 50) -> int:
        """
        Welch's method for transient detection
        Returns: index where transient phase ends
        """
        if len(data) < window_size * 2:
            return len(data) // 4  # Default to 25% if not enough data

        n = len(data)
        moving_avg = []

        # Calculate moving average
        for i in range(window_size, n):
            moving_avg.append(np.mean(data[i-window_size:i]))

        # Find point of stabilization (minimum variance in moving average)
        variances = []
        for i in range(50, len(moving_avg) - 50):
            var = np.var(moving_avg[i:i+50])
            variances.append(var)

        if not variances:
            return len(data) // 4

        min_var_idx = np.argmin(variances) + 50 + window_size
        return min(min_var_idx, n // 2)  # Cap at 50% of data

    @staticmethod
    def plot_transient_detection(data: List[float], transient_end: int,
                                 title: str = "Transient Detection"):
        """Visualize transient detection"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Full data plot with transient marker
        axes[0].plot(data, alpha=0.7, linewidth=1.5, color='#2E86AB', label='Raw Data')
        axes[0].axvline(x=transient_end, color='red', linestyle='--', linewidth=2,
                       label=f'Transient End (sample {transient_end})')
        axes[0].axvspan(0, transient_end, alpha=0.2, color='red', label='Transient Phase')
        axes[0].axvspan(transient_end, len(data), alpha=0.2, color='green', label='Steady State')
        axes[0].set_xlabel('Sample Index', fontsize=11)
        axes[0].set_ylabel('Metric Value', fontsize=11)
        axes[0].set_title('Complete Time Series', fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Moving average plot
        window = 50
        if len(data) >= window:
            moving_avg = [np.mean(data[max(0, i-window):i+1]) for i in range(len(data))]
            axes[1].plot(moving_avg, linewidth=2, color='#F18F01', label='Moving Average')
            axes[1].axvline(x=transient_end, color='red', linestyle='--', linewidth=2,
                           label=f'Detected Transient End')

            # Add mean lines
            transient_mean = np.mean(data[:transient_end])
            steady_mean = np.mean(data[transient_end:])
            axes[1].axhline(y=transient_mean, color='red', linestyle=':', alpha=0.7,
                           label=f'Transient Mean: {transient_mean:.2f}')
            axes[1].axhline(y=steady_mean, color='green', linestyle=':', alpha=0.7,
                           label=f'Steady Mean: {steady_mean:.2f}')

            axes[1].set_xlabel('Sample Index', fontsize=11)
            axes[1].set_ylabel('Moving Average', fontsize=11)
            axes[1].set_title(f'Moving Average (window={window})', fontweight='bold')
            axes[1].legend(loc='upper right', fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

class ConfidenceIntervalCalculator:
    """
    Calculate confidence intervals for simulation output
    """

    @staticmethod
    def independent_replications(replications: List[float],
                                confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate CI using independent replications method
        Returns: (mean, lower_bound, upper_bound)
        """
        n = len(replications)
        mean = np.mean(replications)
        std = np.std(replications, ddof=1)

        # t-distribution for small samples
        t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
        margin = t_value * std / np.sqrt(n)

        return mean, mean - margin, mean + margin

    @staticmethod
    def batch_means(data: List[float], num_batches: int = 10,
                   confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate CI using batch means method
        Returns: (mean, lower_bound, upper_bound)
        """
        batch_size = len(data) // num_batches
        batch_means = []

        for i in range(num_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            batch_means.append(np.mean(batch))

        return ConfidenceIntervalCalculator.independent_replications(
            batch_means, confidence_level)

# ============================================================================
# SCENARIO TESTING & ANALYSIS
# ============================================================================

def test_mm1_queue(arrival_rate: float, service_rate: float,
                   capacity: int, initial_state: str, sim_time: float = 10000):
    """
    Test transient detection on M/M/1 queue (Lab 4 Scenario 1)
    """
    print(f"\n{'='*60}")
    print(f"M/M/1 Queue Test: λ={arrival_rate}, μ={service_rate}")
    print(f"Initial state: {initial_state}, Capacity: {capacity}")
    print(f"{'='*60}")

    # Simulate queue length over time
    queue_lengths = []
    time = 0
    queue_size = capacity if initial_state == "full" else 0

    while time < sim_time:
        queue_lengths.append(queue_size)

        # Next event time
        next_arrival = np.random.exponential(1/arrival_rate)
        next_departure = np.random.exponential(1/service_rate) if queue_size > 0 else float('inf')

        if next_arrival < next_departure:
            time += next_arrival
            if queue_size < capacity:
                queue_size += 1
        else:
            time += next_departure
            queue_size -= 1

    # Detect transient
    transient_end = TransientDetector.welch_method(queue_lengths)

    print(f"Transient period: {transient_end} samples")
    print(f"Steady-state mean queue length: {np.mean(queue_lengths[transient_end:]):.2f}")
    print(f"System utilization (ρ): {arrival_rate/service_rate:.2f}")

    # Plot transient detection
    TransientDetector.plot_transient_detection(
        queue_lengths, transient_end,
        f"M/M/1 Queue: λ={arrival_rate}, μ={service_rate}, {initial_state}"
    )
    plt.savefig(f'mm1_queue_{initial_state}_rho_{arrival_rate/service_rate:.1f}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    return queue_lengths, transient_end

def run_car_sharing_experiments():
    """
    Run comprehensive experiments on car-sharing system (Lab 3 & 4)
    """
    print("\n" + "="*80)
    print("CAR-SHARING SYSTEM EXPERIMENTS")
    print("="*80)

    # Experiment 1: Varying fleet size
    print("\n--- Experiment 1: Fleet Size Impact ---")
    fleet_sizes = [50, 100, 150, 200]
    results = []
    availability_rates = []
    avg_waiting_times = []
    utilization_rates = []

    for size in fleet_sizes:
        sim = CarSharingSimulator(
            fleet_size=size,
            user_arrival_rate=60.0,
            random_seed=42  # ENHANCEMENT 1: Controlled seed
        )
        sim.run(simulation_time=1440)  # 24 hours
        metrics = sim.get_performance_metrics()
        results.append(metrics)
        availability_rates.append(metrics['availability_rate'])
        avg_waiting_times.append(metrics['avg_waiting_time'])
        utilization_rates.append(metrics['avg_utilization'])

        print(f"Fleet size {size}: Availability={metrics['availability_rate']:.2%}, "
              f"Avg wait={metrics['avg_waiting_time']:.2f} min, "
              f"Utilization={metrics['avg_utilization']:.2%}")

    # Plot Experiment 1 results
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Impact of Fleet Size on System Performance', fontsize=16, fontweight='bold')

    axes[0].plot(fleet_sizes, availability_rates, marker='o', linewidth=2,
                 markersize=8, color='#2E86AB')
    axes[0].set_xlabel('Fleet Size', fontsize=11)
    axes[0].set_ylabel('Availability Rate', fontsize=11)
    axes[0].set_title('Service Availability vs Fleet Size', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(fleet_sizes, avg_waiting_times, marker='s', linewidth=2,
                 markersize=8, color='#F18F01')
    axes[1].set_xlabel('Fleet Size', fontsize=11)
    axes[1].set_ylabel('Avg Waiting Time (min)', fontsize=11)
    axes[1].set_title('Waiting Time vs Fleet Size', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(fleet_sizes, utilization_rates, marker='^', linewidth=2,
                 markersize=8, color='#A23B72')
    axes[2].set_xlabel('Fleet Size', fontsize=11)
    axes[2].set_ylabel('Utilization Rate', fontsize=11)
    axes[2].set_title('Fleet Utilization vs Fleet Size', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig('fleet_size_impact.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Experiment 2: Multiple replications for CI (with different seeds)
    print("\n--- Experiment 2: Confidence Intervals (10 replications) ---")
    waiting_times = []
    availability_rates_rep = []

    for rep in range(10):
        # ENHANCEMENT 1: Each replication gets unique seed
        sim = CarSharingSimulator(
            fleet_size=100,
            user_arrival_rate=50.0,
            random_seed=42 + rep  # Different seed per replication
        )
        sim.run(simulation_time=1440)
        metrics = sim.get_performance_metrics()
        waiting_times.append(metrics['avg_waiting_time'])
        availability_rates_rep.append(metrics['availability_rate'])
        print(f"  Replication {rep+1}: Wait={metrics['avg_waiting_time']:.2f} min, "
              f"Avail={metrics['availability_rate']:.2%}")

    mean_wait, lower_wait, upper_wait = ConfidenceIntervalCalculator.independent_replications(waiting_times)
    mean_avail, lower_avail, upper_avail = ConfidenceIntervalCalculator.independent_replications(availability_rates_rep)

    print(f"\nAverage waiting time: {mean_wait:.2f} min")
    print(f"95% CI: [{lower_wait:.2f}, {upper_wait:.2f}]")
    print(f"Average availability: {mean_avail:.2%}")
    print(f"95% CI: [{lower_avail:.2%}, {upper_avail:.2%}]")

    # Plot confidence intervals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confidence Intervals from 10 Independent Replications', fontsize=16, fontweight='bold')

    # Waiting time CI
    axes[0].bar(['Waiting Time'], [mean_wait], color='#2E86AB', alpha=0.7, width=0.5)
    axes[0].errorbar(['Waiting Time'], [mean_wait],
                     yerr=[[mean_wait - lower_wait], [upper_wait - mean_wait]],
                     fmt='none', color='black', capsize=10, capthick=2, linewidth=2)
    axes[0].set_ylabel('Time (minutes)', fontsize=11)
    axes[0].set_title('Average Waiting Time with 95% CI', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].text(0, mean_wait + (upper_wait - mean_wait) + 0.5,
                f'[{lower_wait:.2f}, {upper_wait:.2f}]',
                ha='center', fontsize=10, fontweight='bold')

    # Availability rate CI
    axes[1].bar(['Availability'], [mean_avail], color='#6A994E', alpha=0.7, width=0.5)
    axes[1].errorbar(['Availability'], [mean_avail],
                     yerr=[[mean_avail - lower_avail], [upper_avail - mean_avail]],
                     fmt='none', color='black', capsize=10, capthick=2, linewidth=2)
    axes[1].set_ylabel('Rate', fontsize=11)
    axes[1].set_title('Availability Rate with 95% CI', fontweight='bold')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].text(0, mean_avail + (upper_avail - mean_avail) + 0.02,
                f'[{lower_avail:.3f}, {upper_avail:.3f}]',
                ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('confidence_intervals.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Experiment 3: Transient detection and detailed analysis
    print("\n--- Experiment 3: Transient Phase Detection & System Evolution ---")
    sim = CarSharingSimulator(fleet_size=100, user_arrival_rate=50.0, random_seed=42)
    sim.run(simulation_time=2880)  # 48 hours

    if sim.waiting_times:
        transient_end = TransientDetector.welch_method(sim.waiting_times, window_size=30)
        print(f"Transient phase: {transient_end} samples")
        print(f"Transient period: ~{transient_end * 5} minutes")
        print(f"Mean waiting time (with transient): {np.mean(sim.waiting_times):.2f} min")
        print(f"Mean waiting time (without transient): {np.mean(sim.waiting_times[transient_end:]):.2f} min")
        improvement = (np.mean(sim.waiting_times) - np.mean(sim.waiting_times[transient_end:])) / np.mean(sim.waiting_times) * 100
        print(f"Improvement by removing transient: {improvement:.1f}%")

        # Plot transient detection for car-sharing
        TransientDetector.plot_transient_detection(
            sim.waiting_times, transient_end,
            "Car-Sharing System: Transient Phase Detection"
        )
        plt.savefig('car_sharing_transient.png', dpi=150, bbox_inches='tight')
        plt.show()

        # ENHANCEMENT 4: Plot system evolution with transient overlay
        sim.plot_system_evolution(transient_end=transient_end)
        plt.savefig('system_evolution.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Plot car locations
        sim.plot_car_locations()
        plt.savefig('car_locations.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Experiment 4: Arrival rate sensitivity
    print("\n--- Experiment 4: Arrival Rate Sensitivity Analysis ---")
    arrival_rates = [30, 40, 50, 60, 70, 80]
    availability_by_rate = []
    waiting_by_rate = []

    for rate in arrival_rates:
        sim = CarSharingSimulator(fleet_size=100, user_arrival_rate=rate, random_seed=42)
        sim.run(simulation_time=1440)
        metrics = sim.get_performance_metrics()
        availability_by_rate.append(metrics['availability_rate'])
        waiting_by_rate.append(metrics['avg_waiting_time'])
        print(f"Arrival rate {rate}/hr: Availability={metrics['availability_rate']:.2%}, "
              f"Wait={metrics['avg_waiting_time']:.2f} min")

    # Plot arrival rate sensitivity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('System Sensitivity to User Arrival Rate', fontsize=16, fontweight='bold')

    axes[0].plot(arrival_rates, availability_by_rate, marker='o', linewidth=2,
                 markersize=8, color='#2E86AB')
    axes[0].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95% Target')
    axes[0].set_xlabel('Arrival Rate (users/hour)', fontsize=11)
    axes[0].set_ylabel('Availability Rate', fontsize=11)
    axes[0].set_title('Service Availability vs Demand', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(arrival_rates, waiting_by_rate, marker='s', linewidth=2,
                 markersize=8, color='#F18F01')
    axes[1].set_xlabel('Arrival Rate (users/hour)', fontsize=11)
    axes[1].set_ylabel('Avg Waiting Time (min)', fontsize=11)
    axes[1].set_title('Waiting Time vs Demand', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('arrival_rate_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ENHANCEMENT 2: Experiment 5: With vs Without Relocation
    print("\n--- Experiment 5: Impact of Relocation Strategy ---")

    # Without relocation
    sim_no_reloc = CarSharingSimulator(
        fleet_size=100,
        user_arrival_rate=60.0,
        relocation_enabled=False,
        random_seed=42
    )
    sim_no_reloc.run(simulation_time=1440)
    metrics_no_reloc = sim_no_reloc.get_performance_metrics()

    # With relocation
    sim_with_reloc = CarSharingSimulator(
        fleet_size=100,
        user_arrival_rate=60.0,
        relocation_enabled=True,
        relocation_threshold=0.3,
        relocation_check_interval=30.0,
        random_seed=42
    )
    sim_with_reloc.run(simulation_time=1440)
    metrics_with_reloc = sim_with_reloc.get_performance_metrics()

    print(f"\nWithout Relocation:")
    print(f"  Availability: {metrics_no_reloc['availability_rate']:.2%}")
    print(f"  Avg Waiting Time: {metrics_no_reloc['avg_waiting_time']:.2f} min")
    print(f"  Utilization: {metrics_no_reloc['avg_utilization']:.2%}")

    print(f"\nWith Relocation:")
    print(f"  Availability: {metrics_with_reloc['availability_rate']:.2%}")
    print(f"  Avg Waiting Time: {metrics_with_reloc['avg_waiting_time']:.2f} min")
    print(f"  Utilization: {metrics_with_reloc['avg_utilization']:.2%}")
    print(f"  Total Relocations: {metrics_with_reloc['total_relocations']}")

    improvement_avail = (metrics_with_reloc['availability_rate'] - metrics_no_reloc['availability_rate']) * 100
    improvement_wait = (metrics_no_reloc['avg_waiting_time'] - metrics_with_reloc['avg_waiting_time']) / metrics_no_reloc['avg_waiting_time'] * 100

    print(f"\nImprovements with Relocation:")
    print(f"  Availability: +{improvement_avail:.1f} percentage points")
    print(f"  Waiting Time: -{improvement_wait:.1f}%")

    # Comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Impact of Relocation Strategy', fontsize=16, fontweight='bold')

    categories = ['No Relocation', 'With Relocation']
    availability_comparison = [metrics_no_reloc['availability_rate'],
                              metrics_with_reloc['availability_rate']]
    waiting_comparison = [metrics_no_reloc['avg_waiting_time'],
                         metrics_with_reloc['avg_waiting_time']]
    utilization_comparison = [metrics_no_reloc['avg_utilization'],
                             metrics_with_reloc['avg_utilization']]

    axes[0].bar(categories, availability_comparison, color=['#E63946', '#06D6A0'], alpha=0.7)
    axes[0].set_ylabel('Availability Rate', fontsize=11)
    axes[0].set_title('Service Availability', fontweight='bold')
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(categories, waiting_comparison, color=['#E63946', '#06D6A0'], alpha=0.7)
    axes[1].set_ylabel('Avg Waiting Time (min)', fontsize=11)
    axes[1].set_title('Customer Waiting Time', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(categories, utilization_comparison, color=['#E63946', '#06D6A0'], alpha=0.7)
    axes[2].set_ylabel('Utilization Rate', fontsize=11)
    axes[2].set_title('Fleet Utilization', fontweight='bold')
    axes[2].set_ylim([0, 1.05])
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('relocation_impact.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Plot relocation analysis
    if sim_with_reloc.relocation_events:
        sim_with_reloc.plot_relocation_analysis()
        plt.savefig('relocation_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print(" CAR-SHARING SYSTEM SIMULATOR - Labs 2, 3 & 4")
    print(" Enhanced with: Controlled Seeds, Relocation Logic, Transient Overlays")
    print("="*80)

    # Lab 4 Scenario 1: M/M/1 Queue Tests
    print("\n" + "="*80)
    print("LAB 4 - SCENARIO 1: M/M/1 QUEUE ANALYSIS")
    print("="*80)

    # Case 1: ρ = 0.8, initially full
    queue1, trans1 = test_mm1_queue(0.8, 1.0, 1000, "full", 5000)

    # Case 2: λ = 1.2μ, initially empty
    queue2, trans2 = test_mm1_queue(1.2, 1.0, 1000, "empty", 5000)

    # Case 3: λ = μ, initially empty
    queue3, trans3 = test_mm1_queue(1.0, 1.0, 1000, "empty", 5000)

    # Lab 4 Scenario 2: Car-Sharing System
    print("\n" + "="*80)
    print("LAB 4 - SCENARIO 2: CAR-SHARING SYSTEM ANALYSIS")
    print("="*80)
    run_car_sharing_experiments()

    print("\n" + "="*80)
    print(" SIMULATION COMPLETE")
    print(" All plots saved as PNG files in the current directory")
    print("="*80)