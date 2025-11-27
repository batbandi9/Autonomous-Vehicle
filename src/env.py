#!/usr/bin/env python3
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional

# Parking environment with dense reward, converted for PPO (non-GoalEnv)

class ParkingPPOEnv(gym.Env):
    """
    Gymnasium environment for autonomous vehicle parking using PPO.
    The vehicle must park in designated spots while avoiding obstacles.
    The observation space is a flat vector suitable for PPO.
    The action space is continuous with steering and acceleration.
    2D top-down view with Pygame rendering.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Environment setup
        self.render_mode = render_mode
        self.screen_width = 1000
        self.screen_height = 800

        self.reward_pygame = 0
        self.initial_delta_x = 0
        self.initial_delta_y = 0
        
        # Structured Layout
        self.parking_spots = [
            (300.0, 300.0, 2*np.pi/4), 
            (800.0, 200.0, 3*np.pi/4),
            (200.0, 200.0, np.pi/4), 
            #(300.0, 500.0, -3*np.pi/4), 
            #(900.0, 500.0, -np.pi/4),
        ]
    
        # Circle shaped obstacles (kept for future use, but empty)
        self.obstacles = [
           # (450.0, 500.0, 80.0),  # (x, y, radius)
           # (550.0, 400.0, 30.0),  # (x, y, radius) 
        ]
        
        # Vehicle specifications
        self.car_length = 40.0 # 4 meters (10 pixels corresponds to 1 meter)
        self.car_width = 20.0 # 2 meters
        self.max_speed = 5.0 # 5m/s = 18 km/h
        self.max_accel = 1.0 # 1 m/s^2 
        self.max_steering = 0.785  # ~45 degrees, 0.785radian
        self.pixels_per_meter = 10 #  10 pixels = 1 meter
        self.dt = 0.1 # Time step in seconds

        # Simulation
        self.max_episode_steps = 1000
        self.step_count = 0
        
        # State variables
        self.car_x, self.car_y, self.car_angle, self.car_speed = 0.0, 0.0, 0.0, 0.0
        self.target_x, self.target_y, self.target_angle = 0.0, 0.0, 0.0

        # Reward function parameters
        self.reward_weights = np.array([
            0.4,  # position (pixels)
            0.4,  # position (pixels)
            0.2, # angle (radians) - highly weighted
            0, # velocity
        ], dtype=np.float32)

        self.reward_p_norm = 1 # first one linear
        self.success_cost_threshold = 4e-4

        # Pygame renderin4
        self.screen = None
        self.clock = None

        # Action Space: 2D continuous [steering, acceleration]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )
        
        # Observation Space: 7D vector for PPO
        # [delta_x, delta_y, car_speed, car_cos, car_sin, target_cos, target_sin]
        obs_low = np.array([
            -1, # delta_x normalized
            -1, # delta_y normalized
            -1, # delta_angle normalized
            -1,
            -1,
            -1,
            -1. # delta velocity normalized
        ], dtype=np.float32)
        
        obs_high = np.array([
            1, # delta_x normalized
            1, # delta_y normalized
            1, # delta_angle normalized
            1,
            1,
            1,
            1. # delta velocity normalized
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    # Main Gymnasium Functions

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:

        super().reset(seed=seed)
        
        # This cycles through the parking spots
        self._spot_counter = getattr(self, "_spot_counter", 0)
        spot_index = self._spot_counter % len(self.parking_spots)
        self._spot_counter += 1
        self.target_x, self.target_y, self.target_angle = self.parking_spots[spot_index]
        
        # This is the car's starting position

        self.car_y = 700
        self.car_x = 500
        self.last_dist = np.sqrt((self.car_x - self.target_x)**2 + (self.car_y - self.target_y)**2)

        self.car_angle =  np.random.uniform(-np.pi/2 -np.pi/8, -np.pi/2 + np.pi/8)
        self.car_speed = 0.0 
        self.step_count = 0

        self.initial_delta_x = self.screen_width
        self.initial_delta_y = self.screen_height
        
        # This gets the flat observation vector
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_human()
            
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        self.reward_pygame = 0
        steering_action = np.clip(action[0], -1.0, 1.0) * self.max_steering
        acceleration_action = np.clip(action[1], -1.0, 1.0) * self.max_accel
        
        # This stores the action for rendering
        self.last_steering_action = steering_action
        self.last_acceleration_action = acceleration_action
        
        # This updates the car's position
        self._update_physics(steering_action, acceleration_action)
        self.step_count += 1
        
        obs = self._get_obs()
        info = self._get_info()
        
        # This computes the reward based on the current state
        reward = self.compute_reward()
        
        # This is a living penalty to encourage speed
        reward -= 0.01
        
        # This initializes the termination flag
        terminated = False
        
        # This adds a crash penalty
        if self._has_crashed():
            reward -= 5.0  # Make crashing very bad
            terminated = True

        # This adds an obstacle collision penalty
        if self._hit_obstacle():
            reward -= 5.0  # Make hitting obstacle very bad
            terminated = True # For convergence could be False
        
        # This adds a success bonus
        if info["is_success"]:
            reward += 1000.0  # Make success very rewarding
            terminated = True  # Terminate on success

        # This truncates the episode if it's too long
        truncated = False
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        # This is the diagnostic print you requested
        if terminated or truncated:
            # This calculates the proximity cost for logging
            proximity_cost = self._compute_proximity_cost()
            print(f"Episode ended at step {self.step_count}: "
                  f"pos=({self.car_x:.1f},{self.car_y:.1f}), "
                  f"crashed={self._has_crashed()}, "
                  #f"hit_obs={self._hit_obstacle()}, " # Uncomment if you add obstacles
                  f"success={info['is_success']}, "
                  f"proximity_cost={proximity_cost:.4f}, "
                  f"steering={steering_action:.2f}, "
                  f"orientation={self.car_angle:.2f} rad, "
                  f"reward={reward:.3f}")
            
        self.reward_pygame = reward

        if self.render_mode == "human":
            self._render_human()
        

        return obs, reward, terminated, truncated, info

    # PPO Helper Functions

    def _get_obs(self):
        
        # --- 1. Position error in global frame ---
        dx = self.target_x - self.car_x
        dy = self.target_y - self.car_y

        # --- 2. Transform to local frame (car-centric coordinates) ---
        dx_local =  dx * np.cos(self.car_angle) + dy * np.sin(self.car_angle)
        dy_local = -dx * np.sin(self.car_angle) + dy * np.cos(self.car_angle)

        # Normalize by world/screen dimensions
        dx_local /= self.screen_width
        dy_local /= self.screen_height

        # --- 3. Angle toward target position (navigation) ---
        angle_to_target = np.arctan2(dy, dx) - self.car_angle
        angle_to_target = (angle_to_target + np.pi) % (2*np.pi) - np.pi

        # Wrap-safe representation
        cos_target = np.cos(angle_to_target)
        sin_target = np.sin(angle_to_target)

        # --- 4. Final desired orientation error (parking) ---
        theta_final = self.target_angle              # desired parking angle
        delta_theta_final = theta_final - self.car_angle
        delta_theta_final = (delta_theta_final + np.pi) % (2*np.pi) - np.pi

        cos_final = np.cos(delta_theta_final)
        sin_final = np.sin(delta_theta_final)

        # --- 5. Speed normalization ---
        speed_norm = self.car_speed / self.max_speed

        # --- Final observation vector ---
        return np.array([
            dx_local, dy_local,        # relative target position
            cos_target, sin_target,    # direction for navigation
            cos_final, sin_final,      # final orientation alignment
            speed_norm,                # normalized speed
        ], dtype=np.float32)



    def _is_success(self) -> bool:
        """Helper to check if goal is achieved using proximity cost."""
        # This gets the cost from the current state
        proximity_cost = self._compute_proximity_cost()
        return proximity_cost < self.success_cost_threshold

    def compute_reward(self) -> float:
        """
        Computes the dense reward based on proximity cost.
        This function no longer needs arguments.
        """
        # This calculates cost from the current state
        proximity_cost = self._compute_proximity_cost()
        
        # This returns the scaled negative cost
        # reward = (1 - proximity_cost)**5
        # reward = 1-proximity_cost
        reward = 1 - np.cbrt(proximity_cost)
        success = self._is_success()
       # if self.car_speed <= 1e-2 and not success:
       #     reward -= 0.5 # small penalty for being stationary

        return reward

 
    
    def _compute_proximity_cost(self) -> float:
        
        # Position error (Euclidean)
        error_x_raw = self.car_x - self.target_x
        error_y_raw = self.car_y - self.target_y
        error_dist = np.sqrt(error_x_raw**2 + error_y_raw**2)

        error_delta_x = abs(error_x_raw/(self.initial_delta_x + 1e-6))
        error_delta_y = abs(error_y_raw/(self.initial_delta_y + 1e-6))
        
        # Error distance normalized to [0, 0.5 ?] considering max possible distance on the reset function
        max_dist = np.sqrt(self.initial_delta_x**2 + self.initial_delta_y**2)
        error_dist = error_dist / (max_dist + 1e-6 )  
        # Angle error
        achieved_angle = self.car_angle
        desired_angle = self.target_angle
        angle_diff = achieved_angle - desired_angle
        
        # This wraps the angle to [-π, π]
        angle_diff = ((angle_diff + np.pi) % (2*np.pi)) - np.pi
        # error_angle = 1 - np.cos(angle_diff)
        error_angle = abs(angle_diff)/np.pi

        # Velocity error
        error_velocity = abs(self.car_speed) / (self.max_speed + 1e-6)  # Normalize to [0, 1]
        
        # Combine the errors
        # Note: self.reward_weights should now be a 4-element array
        errors = np.array([error_delta_x, error_delta_y, error_angle, error_velocity])
        
        weighted_error_sum = np.dot(errors, self.reward_weights)
        epsilon = 1e-6
        
        # We still use p=0.5 to make it a "perfectionist" agent
        return np.power(weighted_error_sum + epsilon, self.reward_p_norm)
    
    def _get_info(self) -> Dict:
        """Get info dictionary."""
        # This checks for success from the current state
        is_success = self._is_success()
        proximity_cost = self._compute_proximity_cost()
        
        return {
            'is_success': is_success, # This is good for logging
            'proximity_cost': proximity_cost,
            'steps': self.step_count,
            'hit_obstacle': self._hit_obstacle(),
            'hit_boundary': self._has_crashed()
        }

    # Physics and Collision Functions

    def _update_physics(self, steering: float, acceleration: float):
        """Update vehicle physics with kinematic bicycle model."""
        
        # This updates the speed
        self.car_speed += acceleration * self.dt
        self.car_speed = np.clip(self.car_speed, -self.max_speed / 2, self.max_speed)
        
        speed_pixels = self.car_speed * self.pixels_per_meter * self.dt 
        
        # This calculates the turning
        if abs(self.car_speed) > 0.01: 
            L_meters = self.car_length / self.pixels_per_meter
            angular_velocity = (self.car_speed / L_meters) * np.tan(steering)
            self.car_angle += angular_velocity * self.dt
        
        # This wraps the angle
        self.car_angle = ((self.car_angle + np.pi) % (2 * np.pi)) - np.pi
        
        # This updates the position
        self.car_x += speed_pixels * np.cos(self.car_angle)
        self.car_y += speed_pixels * np.sin(self.car_angle)
        
    def _has_crashed(self) -> bool:
        """Check for collision with screen boundaries."""
        half_len = self.car_length / 2
        half_wid = self.car_width / 2
        return (self.car_x <= half_len or
                self.car_x >= self.screen_width - half_len or
                self.car_y <= half_wid or 
                self.car_y >= self.screen_height - half_wid)
    
    def _hit_obstacle(self) -> bool:
        """Check if car collides with any obstacle (circle collision)."""
        # This logic is kept for when you add obstacles
        if not self.obstacles:
            return False
            
        car_center_x = self.car_x
        car_center_y = self.car_y
        
        # This is a simple car radius approximation
        car_radius = np.sqrt((self.car_length / 2)**2 + (self.car_width / 2)**2)
        
        for obs_x, obs_y, obs_radius in self.obstacles:
            # This is the distance between centers
            distance = np.sqrt((car_center_x - obs_x)**2 + (car_center_y - obs_y)**2)
            
            # This checks for collision
            if distance < (car_radius + obs_radius):
                return True
        
        return False

    # Rendering Functions

    def render(self):
        """Main render-dispatch function."""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
            
    def close(self):
        """Close the pygame window."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    def _draw_parking_lot(self):
        """Draws obstacles and parking spots."""
        # This draws the obstacles as red circles
        for obs_x, obs_y, obs_radius in self.obstacles:
            pygame.draw.circle(self.screen, (255, 0, 0), (int(obs_x), int(obs_y)), int(obs_radius))
            pygame.draw.circle(self.screen, (150, 0, 0), (int(obs_x), int(obs_y)), int(obs_radius), 3)

        # Draw parking spots (green ghost vehicles with orientation)
        for (x, y, angle) in self.parking_spots:
            # Draw orientation line for the parking spot
            spot_front_x = x + (self.car_length / 2) * np.cos(angle)
            spot_front_y = y + (self.car_length / 2) * np.sin(angle)
            pygame.draw.line(self.screen, (0, 255, 0), (x, y), (spot_front_x, spot_front_y), 3)
            # Draw ghost vehicle for the parking spot (green)
            self._draw_vehicle(x, y, angle, (0, 0, 255, 100), is_ghost=True)

    def _render_human(self):
        """Render the environment to the screen."""
        # This calculates proximity cost for display
        proximity_cost = self._compute_proximity_cost()
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Parking PPO Env")
            pygame.font.init() 
            
        # This is the background
        self.screen.fill((50, 50, 50))
        
        # This draws the spots and obstacles
        self._draw_parking_lot()
        
        # This draws the target spot (ghost)
        spot_color = (0, 255, 0, 100)
        self._draw_vehicle(self.target_x, self.target_y, self.target_angle, spot_color, is_ghost=True)
        
        # This gets the current info
        info = self._get_info()
        is_parked = info['is_success']
        car_color = (0, 255, 0) if is_parked else (255, 0, 0)
        
        # This draws the actual car
        self._draw_vehicle(self.car_x, self.car_y, self.car_angle, car_color)
        
        # This calculates display info
        dx = self.car_x - self.target_x
        dy = self.car_y - self.target_y
        current_dist = np.sqrt(dx*dx + dy*dy)
        angle_diff = self.car_angle - self.target_angle
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        angle_deg = np.degrees(angle_diff)
        
        # This gets the last action
        steering_angle = getattr(self, 'last_steering_action', 0.0) 
        acceleration = getattr(self, 'last_acceleration_action', 0.0) 
        
        # This displays the text
        font = pygame.font.Font(None, 30)
        info_text = [
            f"Distance: {current_dist:.1f} px",
            f"Angle Diff: {angle_deg:.1f} deg",
            f"Steps: {self.step_count}",
            f"Status: {'PARKED' if is_parked else 'PARKING'}",
            f"Proximity Cost: {proximity_cost:.3f}",
            f"Steering: {steering_angle:.1f} rad/s",
            f"Acceleration: {acceleration:.2f} m/s²",
            f"reward: {self.reward_pygame}",
            ]
        
        for i, text in enumerate(info_text):
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (10, 10 + i * 25))
            
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_vehicle(self, x: float, y: float, angle: float, color: Tuple, is_ghost: bool = False):
        """Helper function to draw a rectangle representing the car."""
        half_length = self.car_length / 2
        half_width = self.car_width / 2
        corners = [
            (-half_length, -half_width), (half_length, -half_width),
            (half_length, half_width), (-half_length, half_width)
        ]
        rotated_corners = []
        for corner_x, corner_y in corners:
            rot_x = corner_x * np.cos(angle) - corner_y * np.sin(angle)
            rot_y = corner_x * np.sin(angle) + corner_y * np.cos(angle)
            rotated_corners.append((x + rot_x, y + rot_y))
            
        if is_ghost:
            # This draws a semi-transparent ghost
            surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            pygame.draw.polygon(surface, color, rotated_corners)
            self.screen.blit(surface, (0,0))
            pygame.draw.polygon(self.screen, (255, 255, 255), rotated_corners, 1) 
       
        else:
            # This draws the solid car
            pygame.draw.polygon(self.screen, color, rotated_corners)
            # This adds a white dot for the front
            front_x = x + half_length * np.cos(angle)
            front_y = y + half_length * np.sin(angle)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(front_x), int(front_y)), 3)
    
    def _render_rgb_array(self):
        """Render to an RGB array for headless training."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            
        self._render_human() 
        rgb_array = pygame.surfarray.array3d(self.screen)
        return np.transpose(rgb_array, axes=(1, 0, 2)) # Convert from (W, H, C) to (H, W, C)
    


# Heuristic agent test function for environment testing
def test_heuristic_agent():
    """Test the environment with a simple agent that drives toward the parking spot."""
    env = ParkingPPOEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        # Heuristic: steer toward the parking spot
        delta_x = env.target_x - env.car_x
        delta_y = env.target_y - env.car_y
        target_angle = np.arctan2(delta_y, delta_x)
        angle_diff = target_angle - env.car_angle
        # Wrap angle to [-pi, pi]
        angle_diff = ((angle_diff + np.pi) % (2 * np.pi)) - np.pi
        # Steering: proportional to angle difference
        steering = np.clip(angle_diff / env.max_steering, -1.0, 1.0)
        # Acceleration: go forward if not close
        dist = np.sqrt(delta_x**2 + delta_y**2)
        acceleration = 1.0 if dist > 20 else 0.0
        action = np.array([steering, acceleration], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()

if __name__ == "__main__":
    # This is a test to run the environment with random actions
    env = ParkingPPOEnv(render_mode="human")
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample() # This takes random actions
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()