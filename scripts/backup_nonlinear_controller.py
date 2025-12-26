#!/usr/bin/env python
"""
| File: nonlinear_controller.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to use the control backends API to create a custom controller 
for the vehicle from scratch and use it to perform a simulation, without using PX4 nor ROS. In this controller, we
provide a quick way of following a given trajectory specified in csv files or track an hard-coded trajectory based
on exponentials! NOTE: This is just an example, to demonstrate the potential of the API. A much more flexible solution
can be achieved
"""

# Imports to be able to log to the terminal with fancy colors
import carb

# Imports from the Pegasus library
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends import Backend

# Auxiliary scipy and numpy modules
import numpy as np
from scipy.spatial.transform import Rotation
from c3bf import cbf_safe_velocity
import time
import matplotlib.pyplot as plt
import os

class NonlinearController(Backend):
    """A nonlinear controller class. It implements a nonlinear controller that allows a vehicle to track
    aggressive trajectories. This controlers is well described in the papers
    
    [1] J. Pinto, B. J. Guerreiro and R. Cunha, "Planning Parcel Relay Manoeuvres for Quadrotors," 
    2021 International Conference on Unmanned Aircraft Systems (ICUAS), Athens, Greece, 2021, 
    pp. 137-145, doi: 10.1109/ICUAS51884.2021.9476757.
    [2] D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors," 
    2011 IEEE International Conference on Robotics and Automation, Shanghai, China, 2011, 
    pp. 2520-2525, doi: 10.1109/ICRA.2011.5980409.
    """

    def __init__(self, 
        trajectory_file: str = None, 
        results_file: str=None, 
        reverse=False, 
        Kp=[10.0, 10.0, 10.0],
        Kd=[8.5, 8.5, 8.5],
        Ki=[1.50, 1.50, 1.50],
        Kr=[3.5, 3.5, 3.5],
        Kw=[0.5, 0.5, 0.5]):

        # The current rotor references [rad/s]
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

        # The current state of the vehicle expressed in the inertial frame (in ENU)
        self.p = np.zeros((3,))                   # The vehicle position
        self.R: Rotation = Rotation.identity()    # The vehicle attitude
        self.w = np.zeros((3,))                   # The angular velocity of the vehicle
        self.v = np.zeros((3,))                   # The linear velocity of the vehicle in the inertial frame
        self.a = np.zeros((3,))                   # The linear acceleration of the vehicle in the inertial frame

        # Define the control gains matrix for the outer-loop
        self.Kp = np.diag(Kp)
        self.Kd = np.diag(Kd)
        self.Ki = np.diag(Ki)
        self.Kr = np.diag(Kr)
        self.Kw = np.diag(Kw)

        self.int = np.array([0.0, 0.0, 0.0])

        # Define the dynamic parameters for the vehicle
        self.m = 1.50        # Mass in Kg
        self.g = 9.81       # The gravity acceleration ms^-2

        # Read the target trajectory from a CSV file inside the trajectories directory
        # if a trajectory is provided. Otherwise, just perform the hard-coded trajectory provided with this controller
        self.index = 0
        if trajectory_file is not None:
            self.trajectory = self.read_trajectory_from_csv(trajectory_file)
            self.max_index, _ = self.trajectory.shape
            self.total_time = 0.0
        # Use the built-in trajectory hard-coded for this controller
        else:
            # Set the initial time for starting when using the built-in trajectory (the time is also used in this case
            # as the parametric value)
            self.total_time = -5.0
            # Signal that we will not used a received trajectory
            self.trajectory = None
            self.max_index = 1

        self.reverse = reverse

        # Auxiliar variable, so that we only start sending motor commands once we get the state of the vehicle
        self.reveived_first_state = False

        # Lists used for analysing performance statistics
        self.results_files = results_file
        self.time_vector = []
        self.desired_position_over_time = []
        self.position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.atittude_error_over_time = []
        self.attitude_rate_error_over_time = []

        # ---- CBF reference state ----
        self.p_ref_cbf = None
        self.cbf_initialized = False
        self.waypoints = [
            np.array([0.0, 0.0, 1.5]),   # WP0: takeoff / start height
            np.array([6.0, 0.0, 1.5])    # WP1: move forward-right
        ]
        self.current_wp_idx = 0
        self.wp_reached_radius = 0.2    # [m] consider waypoint reached
        self.k_wp = 0.8                 # position → nominal velocity gain
        self.v_nominal_max = 1.5


        #Initialization for plots
        self.v_nominal_over_time = []        # 3D nominal velocity
        self.v_safe_over_time = []           # 3D CBF-safe velocity
        self.min_dist_over_time = []         # min distance to any obstacle
        self.rpy_over_time = []              # roll, pitch, yaw (deg)
        self.qp_time_over_time = []          # CBF QP solve time (ms)
        

    def read_trajectory_from_csv(self, file_name: str):
        """Auxiliar method used to read the desired trajectory from a CSV file

        Args:
            file_name (str): A string with the name of the trajectory inside the trajectories directory

        Returns:
            np.ndarray: A numpy matrix with the trajectory desired states over time
        """

        # Read the trajectory to a pandas frame
        return np.flip(np.genfromtxt(file_name, delimiter=','), axis=0)


    def start(self):
        """
        Reset the control and trajectory index
        """
        self.reset_statistics()
        

    def stop(self):
        """
        Stopping the controller. Saving the statistics data for plotting later
        and generating figures + metrics for the report.
        """

        if len(self.time_vector) == 0:
            carb.log_warn("No data collected, skipping statistics and plots.")
            return

        # ---------------------------------
        # Stack logs into numpy arrays
        # ---------------------------------
        t = np.array(self.time_vector)
        p = np.vstack(self.position_over_time)             # [N, 3]
        p_ref = np.vstack(self.desired_position_over_time) # [N, 3]
        ep = np.vstack(self.position_error_over_time)      # [N, 3]
        ev = np.vstack(self.velocity_error_over_time)      # [N, 3]
        eR = np.vstack(self.atittude_error_over_time)      # [N, 3]
        ew = np.vstack(self.attitude_rate_error_over_time) # [N, 3]

        v_nominal = np.vstack(self.v_nominal_over_time) if len(self.v_nominal_over_time) > 0 else None
        v_safe = np.vstack(self.v_safe_over_time) if len(self.v_safe_over_time) > 0 else None
        min_dist = np.array(self.min_dist_over_time) if len(self.min_dist_over_time) > 0 else None
        rpy = np.vstack(self.rpy_over_time) if len(self.rpy_over_time) > 0 else None
        qp_times = np.array(self.qp_time_over_time) if len(self.qp_time_over_time) > 0 else None

        # ---------------------------------
        # (Optional) Save statistics to file
        # ---------------------------------
        if self.results_files is not None:
            statistics = {}
            statistics["time"] = t
            statistics["p"] = p
            statistics["desired_p"] = p_ref
            statistics["ep"] = ep
            statistics["ev"] = ev
            statistics["er"] = eR
            statistics["ew"] = ew
            np.savez(self.results_files, **statistics)
            carb.log_warn("Statistics saved to: " + self.results_files)

        # ---------------------------------
        # 1) Tracking Metrics (for RMSE/MAE table)
        # ---------------------------------
        pos_err = p - p_ref   # error wrt CBF-safe reference
        rmse = np.sqrt(np.mean(pos_err**2, axis=0))   # [RMSE_x, RMSE_y, RMSE_z]
        mae = np.mean(np.linalg.norm(pos_err, axis=1))

        print("\n====== Tracking Metrics (CBF case) ======")
        print(f"RMSE_x: {rmse[0]:.4f} m")
        print(f"RMSE_y: {rmse[1]:.4f} m")
        print(f"RMSE_z: {rmse[2]:.4f} m")
        print(f"MAE_pos_norm: {mae:.4f} m")
        print("========================================\n")

        # ---------------------------------
        # 2) Safety Metrics (for safety table)
        # ---------------------------------
        if min_dist is not None and len(min_dist) > 0:
            min_overall = np.min(min_dist)
            drone_radius = 0.2
            r_obs_max = 0.5
            safe_radius = drone_radius + r_obs_max
            collision_occurred = np.any(min_dist < 0.0)

            print("====== Safety Metrics (CBF case) ======")
            print(f"Minimum distance to any obstacle (over time): {min_overall:.4f} m")
            print(f"Effective safe radius (drone + max obstacle): {safe_radius:.4f} m")
            print(f"Collision occurred? {'YES' if collision_occurred else 'NO'}")
            print("=======================================\n")

        # ---------------------------------
        # 3) Computational Cost (QP time table)
        # ---------------------------------
        if qp_times is not None and len(qp_times) > 0:
            mean_qp = np.mean(qp_times)
            max_qp = np.max(qp_times)
            print("====== C3BF QP Timing (ms) ======")
            print(f"Mean QP solve time: {mean_qp:.3f} ms")
            print(f"Max QP solve time : {max_qp:.3f} ms")
            print("=================================\n")

        # ---------------------------------
        # 4) Generate Plots
        # ---------------------------------

        # (1) 3D Trajectory Plot
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection="3d")
        ax1.plot(p[:, 0], p[:, 1], p[:, 2], label="Actual", linewidth=2)
        ax1.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], "--", label="CBF-Safe Ref", linewidth=2)
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")
        ax1.set_zlabel("Z [m]")
        ax1.set_title("3D Trajectory (CBF)")
        ax1.legend()
        ax1.grid(True)

        # (2) Top-Down (XY) Avoidance Plot
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(p[:, 0], p[:, 1], label="Actual", linewidth=2)
        ax2.plot(p_ref[:, 0], p_ref[:, 1], "--", label="CBF-Safe Ref", linewidth=2)

        # Draw obstacles as circles in XY plane
        obstacles_xy = [
            (np.array([-1.0, -1.7]), 0.5 + 0.2),
            (np.array([-2.3, 0.5]), 0.5+0.2),
        ]
        for center, rad in obstacles_xy:
            circle = plt.Circle(center, rad, fill=False, linestyle="--")
            ax2.add_patch(circle)
            ax2.plot(center[0], center[1], "rx")

        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")
        ax2.set_title("Top-Down View: Obstacle Avoidance")
        ax2.axis("equal")
        ax2.legend()
        ax2.grid(True)

        # (3) Tracking Errors Plot
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(t, pos_err[:, 0], label="e_x")
        ax3.plot(t, pos_err[:, 1], label="e_y")
        ax3.plot(t, pos_err[:, 2], label="e_z")
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Position Error [m]")
        ax3.set_title("Tracking Errors")
        ax3.legend()
        ax3.grid(True)

        # (4) Velocity Profile: Nominal vs CBF-Safe
        if v_nominal is not None and v_safe is not None:
            fig4 = plt.figure()
            ax4 = fig4.add_subplot(111)
            ax4.plot(t, np.linalg.norm(v_nominal, axis=1), label="||v_nominal||")
            ax4.plot(t, np.linalg.norm(v_safe, axis=1), label="||v_safe||")
            ax4.set_xlabel("Time [s]")
            ax4.set_ylabel("Speed [m/s]")
            ax4.set_title("Velocity Profile: Nominal vs C3BF-Safe")
            ax4.legend()
            ax4.grid(True)

        # (5) Distance to Obstacle Over Time
        if min_dist is not None:
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(111)
            ax5.plot(t, min_dist, label="Min distance to any obstacle")
            ax5.axhline(0.0, color="r", linestyle="--", label="Collision boundary")
            ax5.set_xlabel("Time [s]")
            ax5.set_ylabel("Distance [m]")
            ax5.set_title("Distance to Closest Obstacle Over Time")
            ax5.legend()
            ax5.grid(True)

        # (6) Attitude Plot (roll, pitch, yaw)
        if rpy is not None:
            fig6 = plt.figure()
            ax6 = fig6.add_subplot(111)
            ax6.plot(t, rpy[:, 0], label="roll [deg]")
            ax6.plot(t, rpy[:, 1], label="pitch [deg]")
            ax6.plot(t, rpy[:, 2], label="yaw [deg]")
            ax6.set_xlabel("Time [s]")
            ax6.set_ylabel("Angle [deg]")
            ax6.set_title("Attitude Over Time")
            ax6.legend()
            ax6.grid(True)

        # ---------------------------------
        # Save all figures to disk (Isaac uses non-interactive backend)
        # ---------------------------------
        fig_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "figures")
        os.makedirs(fig_dir, exist_ok=True)

        fig1.savefig(os.path.join(fig_dir, "traj_3d.png"), dpi=300, bbox_inches="tight")
        fig2.savefig(os.path.join(fig_dir, "topdown_avoidance.png"), dpi=300, bbox_inches="tight")
        fig3.savefig(os.path.join(fig_dir, "tracking_errors.png"), dpi=300, bbox_inches="tight")
        if v_nominal is not None and v_safe is not None:
            fig4.savefig(os.path.join(fig_dir, "velocity_profile.png"), dpi=300, bbox_inches="tight")
        if min_dist is not None:
            fig5.savefig(os.path.join(fig_dir, "distance_to_obstacle.png"), dpi=300, bbox_inches="tight")
        if rpy is not None:
            fig6.savefig(os.path.join(fig_dir, "attitude_rpy.png"), dpi=300, bbox_inches="tight")

        carb.log_warn(f"Saved figures to: {fig_dir}")

        # No plt.show() in Isaac Sim (non-interactive backend)
        # plt.show()

        # Finally, reset if needed
        self.reset_statistics()


    def update_sensor(self, sensor_type: str, data):
        """
        Do nothing. For now ignore all the sensor data and just use the state directly for demonstration purposes. 
        This is a callback that is called at every physics step.

        Args:
            sensor_type (str): The name of the sensor providing the data
            data (dict): A dictionary that contains the data produced by the sensor
        """
        pass

    def update_state(self, state: State):
        """
        Method that updates the current state of the vehicle. This is a callback that is called at every physics step

        Args:
            state (State): The current state of the vehicle.
        """
        self.p = state.position
        self.R = Rotation.from_quat(state.attitude)
        self.w = state.angular_velocity
        self.v = state.linear_velocity

        self.reveived_first_state = True

    def input_reference(self):
        """
        Method that is used to return the latest target angular velocities to be applied to the vehicle

        Returns:
            A list with the target angular velocities for each individual rotor of the vehicle
        """
        return self.input_ref

    def update(self, dt: float):
        """Method that implements the nonlinear control law and updates the target angular velocities for each rotor. 
        This method will be called by the simulation on every physics step

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """
        
        if self.reveived_first_state == False:
            return

        # -------------------------------------------------
        # Update the references for the controller to track
        # -------------------------------------------------
        self.total_time += dt
        

        # Check if we need to update to the next trajectory index
        if self.trajectory is not None:
            if self.index < self.max_index - 1 and self.total_time >= self.trajectory[self.index + 1, 0]:
                self.index += 1

        # Update using an external trajectory
        if self.trajectory is not None:
            # the target positions [m], velocity [m/s], accelerations [m/s^2], jerk [m/s^3], yaw-angle [rad], yaw-rate [rad/s]
            p_ref = np.array([self.trajectory[self.index, 1], self.trajectory[self.index, 2], self.trajectory[self.index, 3]])
            v_ref = np.array([self.trajectory[self.index, 4], self.trajectory[self.index, 5], self.trajectory[self.index, 6]])
            a_ref = np.array([self.trajectory[self.index, 7], self.trajectory[self.index, 8], self.trajectory[self.index, 9]])
            j_ref = np.array([self.trajectory[self.index, 10], self.trajectory[self.index, 11], self.trajectory[self.index, 12]])
            yaw_ref = self.trajectory[self.index, 13]
            yaw_rate_ref = self.trajectory[self.index, 14]
            
        # Or update the reference using the built-in trajectory
        else:
            # Current waypoint goal
            goal = self.waypoints[self.current_wp_idx]
            pos_err_wp = goal - self.p
            dist_wp = np.linalg.norm(pos_err_wp)

            # If close enough, switch to next waypoint (if any)
            if dist_wp < self.wp_reached_radius and self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1
                goal = self.waypoints[self.current_wp_idx]
                pos_err_wp = goal - self.p
                dist_wp = np.linalg.norm(pos_err_wp)

            # Position reference is just the waypoint position
            p_ref = goal.copy()

            # Nominal velocity towards waypoint (proportional to position error)
            if dist_wp > 1e-3:
                v_ref = self.k_wp * pos_err_wp
                # Saturate nominal planning speed
                v_norm = np.linalg.norm(v_ref)
                if v_norm > self.v_nominal_max:
                    v_ref = v_ref * (self.v_nominal_max / v_norm)
            else:
                v_ref = np.zeros(3)

            # For now, no feedforward accel/jerk for this simple nav
            a_ref = np.zeros(3)
            j_ref = np.zeros(3)
            yaw_ref = 0.0
            yaw_rate_ref = 0.0


        # -------------------------------------------------
        # Start the controller implementation
        # -------------------------------------------------

        #Paramers for cbf
        p =self.p
        v_nominal = v_ref

        #Obstacles - sphere
        obstacles = [
            {
                "shape": "cylinder_z",                     
                "c": np.array([-1.0, -1.4, 1.2]),          
                "z_min": 0.0,                              
                "z_max": 2.0,                              
                "c_dot": np.zeros(3),
                "r_obs": 0.5                               
            },

            {
                "shape": "cylinder_z",
                "c": np.array([-2.3, 0.5, 1.2]),
                "z_min": 0.0,
                "z_max": 2.0,
                "c_dot": np.zeros(3),
                "r_obs": 0.5
            }
        ]

        #parameters
        params = {
            "drone_radius": 0.2,
            "gamma": 1.0,
            "v_max": 2.0,
            "safety_margin": 0.1
        }

        # Time the CBF QP (for computational cost table)
        t0 = time.perf_counter()
        # v_safe = cbf_safe_velocity(p, v_nominal, obstacles, params)
        qp_time_ms = (time.perf_counter() - t0) * 1e3  # [ms]

        #Use v_safe instead of v_ref
        v_safe = cbf_safe_velocity(p, v_nominal, obstacles, params)
        if not self.cbf_initialized:
            self.p_ref_cbf = p_ref.copy()
            self.cbf_initialized = True

        # Integrate safe velocity to get a CBF-safe position reference
        self.p_ref_cbf = self.p_ref_cbf + v_safe * dt

         # ---------------------------
        # Extra logs for analysis
        # ---------------------------
        # Log velocities
        self.v_nominal_over_time.append(v_nominal.copy())
        self.v_safe_over_time.append(v_safe.copy())

        # Log min distance to obstacles
        min_dist = np.inf
        for obs in obstacles:
            center = obs["c"]
            r_obs = obs["r_obs"]
            shape = obs.get("shape", "sphere")

            if shape == "sphere":
                d_raw = np.linalg.norm(p - center) - (params["drone_radius"] + r_obs)

            elif shape == "cylinder_z":
                z_min = obs.get("z_min", -np.inf)
                z_max = obs.get("z_max",  np.inf)

                # If outside vertical extent, treat as "far"
                if p[2] < (z_min - params["drone_radius"]) or p[2] > (z_max + params["drone_radius"]):
                    d_raw = np.inf
                else:
                    delta_xy = p[:2] - center[:2]
                    d_raw = np.linalg.norm(delta_xy) - (params["drone_radius"] + r_obs)

            else:
                # unknown shape → ignore
                d_raw = np.inf

            if d_raw < min_dist:
                min_dist = d_raw

        self.min_dist_over_time.append(min_dist)

        # Log attitude as roll, pitch, yaw [deg]
        rpy = self.R.as_euler("xyz", degrees=True)
        self.rpy_over_time.append(rpy)

        # Log QP solve time
        self.qp_time_over_time.append(qp_time_ms)

        # Compute the tracking errors
        # ep = self.p - p_ref
        # ev = self.v - v_ref
        ep = self.p - self.p_ref_cbf #cbf values
        ev = self.v - v_safe #values
        self.int = self.int +  (ep * dt)
        ei = self.int

        # Compute F_des term
        F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ ei) + np.array([0.0, 0.0, self.m * self.g]) + (self.m * a_ref)

        # Get the current axis Z_B (given by the last column of the rotation matrix)
        Z_B = self.R.as_matrix()[:,2]

        # Get the desired total thrust in Z_B direction (u_1)
        u_1 = F_des @ Z_B

        # Compute the desired body-frame axis Z_b
        Z_b_des = F_des / np.linalg.norm(F_des)

        # Compute X_C_des 
        X_c_des = np.array([np.cos(yaw_ref), np.sin(yaw_ref), 0.0])

        # Compute Y_b_des
        Z_b_cross_X_c = np.cross(Z_b_des, X_c_des)
        Y_b_des = Z_b_cross_X_c / np.linalg.norm(Z_b_cross_X_c)

        # Compute X_b_des
        X_b_des = np.cross(Y_b_des, Z_b_des)

        # Compute the desired rotation R_des = [X_b_des | Y_b_des | Z_b_des]
        R_des = np.c_[X_b_des, Y_b_des, Z_b_des]
        R = self.R.as_matrix()

        # Compute the rotation error
        e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))

        # Compute an approximation of the current vehicle acceleration in the inertial frame (since we cannot measure it directly)
        self.a = (u_1 * Z_B) / self.m - np.array([0.0, 0.0, self.g])

        # Compute the desired angular velocity by projecting the angular velocity in the Xb-Yb plane
        # projection of angular velocity on xB − yB plane
        # see eqn (7) from [2].
        hw = (self.m / u_1) * (j_ref - np.dot(Z_b_des, j_ref) * Z_b_des) 
        
        # desired angular velocity
        w_des = np.array([-np.dot(hw, Y_b_des), 
                           np.dot(hw, X_b_des), 
                           yaw_rate_ref * Z_b_des[2]])

        # Compute the angular velocity error
        e_w = self.w - w_des

        # Compute the torques to apply on the rigid body
        tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

        # Use the allocation matrix provided by the Multirotor vehicle to convert the desired force and torque
        # to angular velocity [rad/s] references to give to each rotor
        if self.vehicle:
            self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)

        # ----------------------------
        # Statistics to save for later
        # ----------------------------
        self.time_vector.append(self.total_time)
        self.position_over_time.append(self.p)
        self.desired_position_over_time.append(self.p_ref_cbf)
        self.position_error_over_time.append(ep)
        self.velocity_error_over_time.append(ev)
        self.atittude_error_over_time.append(e_R)
        self.attitude_rate_error_over_time.append(e_w)

        

    @staticmethod
    def vee(S):
        """Auxiliary function that computes the 'v' map which takes elements from so(3) to R^3.

        Args:
            S (np.array): A matrix in so(3)
        """
        return np.array([-S[1,2], S[0,2], -S[0,1]])
    
    def reset_statistics(self):

        self.index = 0
        # If we received an external trajectory, reset the time to 0.0
        if self.trajectory is not None:
            self.total_time = 0.0
        # if using the internal trajectory, make the parametric value start at -5.0
        else:
            self.total_time = -5.0

        # Reset the lists used for analysing performance statistics
        self.time_vector = []
        self.desired_position_over_time = []
        self.position_over_time = []
        self.position_error_over_time = []
        self.velocity_error_over_time = []
        self.atittude_error_over_time = []
        self.attitude_rate_error_over_time = []
        # Reset CBF reference
        self.p_ref_cbf = None
        self.cbf_initialized = False


        #Logging for plots
        self.v_nominal_over_time = []
        self.v_safe_over_time = []
        self.min_dist_over_time = []
        self.rpy_over_time = []
        self.qp_time_over_time = []

    # ---------------------------------------------------
    # Definition of an exponential trajectory for example
    # This can be used as a reference if not trajectory file is passed
    # as an argument to the constructor of this class
    # ---------------------------------------------------

    def pd(self, t, s, reverse=False):
        """The desired position of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A 3x1 array with the x, y ,z desired [m]
        """

        x = t
        z = 1 / s * np.exp(-0.5 * np.power(t/s, 2)) + 1.0
        y = 1 / s * np.exp(-0.5 * np.power(t/s, 2))

        if reverse == True:
            y = -1 / s * np.exp(-0.5 * np.power(t/s, 2)) + 4.5

        return np.array([x,y,z])

    def d_pd(self, t, s, reverse=False):
        """The desired velocity of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A 3x1 array with the d_x, d_y ,d_z desired [m/s]
        """

        x = 1.0
        y = -(t * np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,3)
        z = -(t * np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,3)

        if reverse == True:
            y = (t * np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,3)

        return np.array([x,y,z])

    def dd_pd(self, t, s, reverse=False):
        """The desired acceleration of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A 3x1 array with the dd_x, dd_y ,dd_z desired [m/s^2]
        """

        x = 0.0
        y = (np.power(t,2)*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,5) - np.exp(-np.power(t,2)/(2*np.power(s,2)))/np.power(s,3)
        z = (np.power(t,2)*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,5) - np.exp(-np.power(t,2)/(2*np.power(s,2)))/np.power(s,3)

        if reverse == True:
            y = np.exp(-np.power(t,2)/(2*np.power(s,2)))/np.power(s,3) - (np.power(t,2)*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,5)

        return np.array([x,y,z])

    def ddd_pd(self, t, s, reverse=False):
        """The desired jerk of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A 3x1 array with the ddd_x, ddd_y ,ddd_z desired [m/s^3]
        """
        x = 0.0
        y = (3*t*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,5) - (np.power(t,3)*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,7)
        z = (3*t*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,5) - (np.power(t,3)*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,7)

        if reverse == True:
            y = (np.power(t,3)*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,7) - (3*t*np.exp(-np.power(t,2)/(2*np.power(s,2))))/np.power(s,5)

        return np.array([x,y,z])

    def yaw_d(self, t, s):
        """The desired yaw of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A float with the desired yaw in rad
        """
        return 0.0
    
    def d_yaw_d(self, t, s):
        """The desired yaw_rate of the built-in trajectory

        Args:
            t (float): The parametric value that guides the equation
            s (float): How steep and agressive the curve is
            reverse (bool, optional): Choose whether we want to flip the curve (so that we can have 2 drones almost touching). Defaults to False.

        Returns:
            np.ndarray: A float with the desired yaw_rate in rad/s
        """
        return 0.0
    
    def reset(self):
        """
        Method that when implemented, should handle the reset of the vehicle simulation to its original state
        """
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        """
        For this demo we do not care about graphical sensors such as camera, therefore we can ignore this callback
        """
        pass