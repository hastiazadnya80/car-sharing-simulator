# 🚗 Car-Sharing System Simulator

A **Discrete Event Simulation (DES)** of a car-sharing system, built as part of a simulation course (Labs 2, 3 & 4) at Politecnico di Torino. The simulator models user arrivals, trip assignments, fleet management, and optional car relocation strategies — with full statistical analysis including confidence intervals and transient phase detection.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Experiments](#experiments)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Requirements](#requirements)

---

## Overview

This simulator models a **free-floating car-sharing service** operating over a 2D spatial area. Users arrive randomly (Poisson process), request a car, and are assigned the nearest available vehicle. The simulation tracks waiting times, fleet availability, and utilization rates over configurable time periods.

The project covers three simulation labs:
- **Lab 2** — Model Design: core classes, event types, performance metrics
- **Lab 3** — Implementation: full DES engine with FES (Future Event Set)
- **Lab 4** — Analysis: transient detection, confidence intervals, sensitivity analysis

---

## Features

- ✅ **Discrete Event Simulation** using a priority-queue Future Event Set (FES)
- ✅ **Poisson user arrivals** with exponential inter-arrival times
- ✅ **Nearest-car assignment** based on Euclidean distance
- ✅ **Zone-based spatial modeling** for demand/supply tracking
- ✅ **Car relocation strategy** to rebalance fleet across zones
- ✅ **Transient phase detection** using Welch's method
- ✅ **Confidence interval estimation** via independent replications and batch means
- ✅ **M/M/1 queue validation** for theoretical verification
- ✅ **Reproducible experiments** with controlled random seeds
- ✅ **Rich visualizations** using Matplotlib and Seaborn

---

## System Architecture

```
CarSharingSimulator
│
├── Events (FES - Priority Queue)
│   ├── USER_ARRIVAL
│   ├── TRIP_START / TRIP_END
│   └── RELOCATION_START / RELOCATION_END
│
├── Entities
│   ├── Car         — location, availability, trip history
│   ├── UserRequest — origin, destination, waiting time
│   └── Zone        — spatial demand/supply tracking
│
├── Analysis
│   ├── TransientDetector        — Welch's method
│   └── ConfidenceIntervalCalculator — independent replications & batch means
│
└── Experiments
    ├── Fleet size impact
    ├── Confidence intervals (10 replications)
    ├── Transient detection
    ├── Arrival rate sensitivity
    └── Relocation strategy comparison
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/hastiazadnya80/car-sharing-simulator.git
cd car-sharing-simulator
```

### 2. Install dependencies

```bash
pip install numpy matplotlib scipy seaborn
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the full simulation

```bash
python simulation-lab.py
```

This runs all experiments automatically and saves plots as PNG files in the current directory.

### Run a custom simulation

```python
from simulation-lab import CarSharingSimulator

sim = CarSharingSimulator(
    fleet_size=100,
    area_size=10.0,
    user_arrival_rate=50.0,    # users per hour
    avg_trip_duration=20.0,    # minutes
    relocation_enabled=True,
    relocation_threshold=0.3,
    random_seed=42
)

sim.run(simulation_time=1440)  # 24 hours in minutes
metrics = sim.get_performance_metrics()
print(metrics)
```

---

## Experiments

The simulation runs **5 experiments** automatically:

| # | Experiment | Description |
|---|---|---|
| 1 | Fleet Size Impact | Compares availability & waiting time for fleet sizes 50–200 |
| 2 | Confidence Intervals | 10 independent replications with 95% CI |
| 3 | Transient Detection | Welch's method on 48-hour simulation |
| 4 | Arrival Rate Sensitivity | Tests rates from 30 to 80 users/hour |
| 5 | Relocation Strategy | Compares system performance with and without relocation |

### Output plots saved:

- `fleet_size_impact.png`
- `confidence_intervals.png`
- `car_sharing_transient.png`
- `system_evolution.png`
- `car_locations.png`
- `arrival_rate_sensitivity.png`
- `relocation_impact.png`
- `relocation_analysis.png`
- `mm1_queue_*.png`

---

## Performance Metrics

| Metric | Description |
|---|---|
| **Availability Rate** | Fraction of time at least one car is available |
| **Avg Waiting Time** | Average time for a car to reach the user (minutes) |
| **Fleet Utilization** | Fraction of cars in active use |
| **Service Rate** | Percentage of requests successfully fulfilled |
| **Total Relocations** | Number of proactive car relocations performed |

---

## Project Structure

```
car-sharing-simulator/
│
├── simulation-lab.py       # Main simulation file
├── README.md               # Project documentation
├── .gitignore              # Python gitignore
└── LICENSE                 # MIT License
```

---

## Requirements

```
numpy
matplotlib
scipy
seaborn
```

Python version: **3.8+**

---

## Author

**Hasti Azadnia**  
MSc Data Science & Engineering — Politecnico di Torino  
[GitHub](https://github.com/hastiazadnya80)

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
