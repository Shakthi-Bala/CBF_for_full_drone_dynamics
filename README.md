# Safety-Critical Quadrotor Control using C3BF in Isaac Sim (Pegasus)

This repository presents a **safety-aware quadrotor control framework** that integrates a **Collision Cone Control Barrier Function (C3BF)** with a **nonlinear geometric controller**, evaluated entirely in **NVIDIA Isaac Sim** using the **Pegasus simulator**.

The project demonstrates how **real-time safety guarantees** can be layered on top of an **aggressive nominal controller** *without trajectory replanning*, using **velocity-level CBF filtering**.

---

## 📌 Key Contributions

- Nonlinear **position–velocity–attitude cascaded controller** for quadrotor flight  
- **Velocity-level C3BF safety filter** for real-time obstacle avoidance  
- Support for:
  - **Spherical obstacles**
  - **Projection-based (cylindrical / wall) obstacles**
- Evaluation on **aggressive minimum-snap trajectories**
- Extensive **quantitative plots**, **videos**, and a **detailed technical report**

---

## 🧱 Project Structure
```bash
.
├── Baseline vs CBF/               # Side-by-side comparison videos (Git LFS)
├── CBF Test Scenarios/            # Individual scenario videos
├── figures/                       # Generated plots (tracking, safety, velocity, etc.)
├── isaac_sim_pegasus_scripts/     # Isaac Sim + Pegasus launch and setup scripts
├── scripts/
│   ├── nonlinear_controller.py    # Nonlinear controller + C3BF integration
│   ├── c3bf_sphere.py             # C3BF for spherical obstacles
│   ├── c3bf_projection.py         # C3BF for projection / cylindrical obstacles
├── trajectory/
│   └── pitch_relay_90_degree.csv  # Aggressive minimum-snap trajectory
├── Control_advanced_track_report.pdf
├── README.md
└── .gitattributes                 # Git LFS tracking for videos
```
## 🛠 Requirements
### Software
- NVIDIA Isaac Sim 5.1.0
- Pegasus Simulator (Isaac extension)
- Python ≥ 3.8
### Python Packages
```bash
pip install numpy scipy matplotlib cvxpy
```
⚠️ Note: Isaac Sim uses a non-interactive matplotlib backend — plots are saved to disk, not displayed.

## 🚁 Simulation Platform
- Simulator: NVIDIA Isaac Sim + Pegasus
- Vehicle: Iris quadrotor
- Controller: Nonlinear geometric controller (SE(3)-based)
- Safety Layer: Collision Cone Control Barrier Function (C3BF)
- Hardware (used for testing): Alienware M16 R2, RTX 4070

## 📈 Trajectory
The trajectory/ folder contains a minimum-snap trajectory:
```bash
pitch_relay_90_degree.csv
```
This trajectory is intentionally aggressive (large pitch excursions) to:
- Stress-test controller stability
- Evaluate C3BF behavior near safety boundaries

## 🧠 Control Architecture
Trajectory / Waypoints → C3BF Safety Filter (Velocity-Level QP) → Nonlinear Position–Velocity–Attitude Controller → Quadrotor Dynamics (Pegasus) → State Feedback (World Frame)

- The nominal controller handles tracking performance.
- The C3BF modifies only the velocity reference, enforcing safety constraints.
- No global replanning or trajectory reshaping is performed.

## 🛑 Obstacle Modeling
### Supported Obstacle Types
1. Spherical Obstacles
  - Used for standard C3BF collision cones
  - Defined by center and radius
2. Cylindrical / Wall Obstacles
  - Handled via projection-based CBF
  - Reduced to closest-point constraints
Obstacle definitions are currently hard-coded in the controller for controlled evaluation.

## ▶️ How to Run
1. Launch Isaac Sim
```bash
isaac_run
```
2. Run the Pegasus Simulation Script
From the Isaac Sim Python environment:
```bash
isaac_run isaac_sim_pegasus_scripts/your_launch_script.py
```
3. Select Controller Mode
Inside nonlinear_controller.py, choose:
- Baseline (No CBF)
- C3BF – Sphere Case
- C3BF – Projection Case

## 📊 Results and Evaluation
For each scenario, the following plots are generated:
- 3D trajectory
- Top-down (XY) view
- Tracking errors (x, y, z)
- Velocity profile (nominal vs safe)
- Distance to closest obstacle
- Attitude (roll, pitch, yaw)

### 📌 Total plots:
- 24 (Baseline) + 24 (C3BF)
- Included in figures/ and the report

## 🎥 Videos
- Baseline vs CBF comparisons stored using Git LFS
- Includes:
    - **Head-on collision cases (baseline)**
    - **Safe deviation cases (C3BF)**
### 📁 See:
```bash
Baseline vs CBF/
CBF Test Scenarios/
```

## 📄 Technical Report
The full mathematical formulation, controller design, C3BF derivation, results, and discussion are documented in:
```bash
Control_advanced_track_report.pdf
```
The report includes:
- C3BF math and constraints
- Controller equations
- RMSE / MAE / safety metrics
- Failure case analysis
- Limitations and future work

## ⚠️ Limitations
- Obstacles are static and manually specified
- No onboard perception or uncertainty modeling
- Safety is kinematic (velocity-level)
- Extremely close obstacles may exceed recoverable set
These are discussed in detail in the report.

## 🔮 Future Work
- Dynamic obstacles and multi-agent avoidance
- Perception-driven obstacle detection (RGB-D)
- Higher-order (acceleration-level) CBFs
- Integration with global planners (RRT*, minimum-snap + CBF)

## 📚 References
- Lee et al., Geometric Tracking Control of a Quadrotor UAV on SE(3)
- Tayal et al., Collision Cone Control Barrier Functions, arXiv 2024
- Mellinger & Kumar, Minimum Snap Trajectory Generation, ICRA 2011

## 👤 Author
Shakthi Bala
Robotics Engineering
Control, Planning, and Safety-Critical Autonomy
