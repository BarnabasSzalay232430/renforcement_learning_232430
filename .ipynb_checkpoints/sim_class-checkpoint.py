import pybullet as p
import time
import pybullet_data
import math
import logging
import os
import random
from pid_class import PIDController  # Import PIDController

class Simulation:
    def __init__(self, num_agents, render=False, rgb_array=False, pid_params=None):
        """
        Initialize the simulation class.

        Args:
        - num_agents: Number of robots in the simulation.
        - render: Whether to render the simulation visually.
        - rgb_array: Whether to capture RGB frames.
        - pid_params: A dictionary containing PID parameters for all robots.
          Format: {
              'x': {'Kp': 1.0, 'Ki': 0.0, 'Kd': 0.0},
              'y': {'Kp': 1.0, 'Ki': 0.0, 'Kd': 0.0},
              'z': {'Kp': 1.0, 'Ki': 0.0, 'Kd': 0.0}
          }
        """
        self.render = render
        self.rgb_array = rgb_array
        self.pid_params = pid_params if pid_params else {
            'x': {'Kp': 1.0, 'Ki': 0.0, 'Kd': 0.0},
            'y': {'Kp': 1.0, 'Ki': 0.0, 'Kd': 0.0},
            'z': {'Kp': 1.0, 'Ki': 0.0, 'Kd': 0.0}
        }

        mode = p.GUI if render else p.DIRECT
        self.physicsClient = p.connect(mode)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Load textures
        texture_list = os.listdir("textures")
        random_texture = random.choice(texture_list[:-1])
        random_texture_index = texture_list.index(random_texture)
        self.plate_image_path = f'textures/_plates/{os.listdir("textures/_plates")[random_texture_index]}'
        self.textureId = p.loadTexture(f'textures/{random_texture}')

        # Set camera parameters
        cameraDistance = 1.1 * (math.ceil((num_agents) ** 0.3))
        cameraYaw = 90
        cameraPitch = -35
        cameraTargetPosition = [-0.2, -(math.ceil(num_agents ** 0.5) / 2) + 0.5, 0.1]
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTargetPosition)

        self.baseplaneId = p.loadURDF("plane.urdf")

        # Define the pipette offset
        self.pipette_offset = [0.073, 0.0895, 0.0895]
        self.pipette_positions = {}

        # Create robots
        self.pid_controllers = []  # Initialize PID controllers
        self.create_robots(num_agents)

        # Initialize additional attributes
        self.sphereIds = []
        self.droplet_positions = {}

    def create_robots(self, num_agents):
        spacing = 1
        grid_size = math.ceil(num_agents ** 0.5)
        self.robotIds = []
        self.specimenIds = []
        agent_count = 0

        for i in range(grid_size):
            for j in range(grid_size):
                if agent_count < num_agents:
                    position = [-spacing * i, -spacing * j, 0.03]
                    robotId = p.loadURDF("ot_2_simulation_v6.urdf", position, [0, 0, 0, 1],
                                        flags=p.URDF_USE_INERTIA_FROM_FILE)

                    # Create a fixed constraint between robot and plane
                    start_position, start_orientation = p.getBasePositionAndOrientation(robotId)
                    p.createConstraint(parentBodyUniqueId=robotId,
                                        parentLinkIndex=-1,
                                        childBodyUniqueId=-1,
                                        childLinkIndex=-1,
                                        jointType=p.JOINT_FIXED,
                                        jointAxis=[0, 0, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=start_position,
                                        childFrameOrientation=start_orientation)

                    # Load specimen with offset
                    offset = [0.1827 - 0.00005, 0.163 - 0.026, 0.057]
                    position_with_offset = [position[0] + offset[0], position[1] + offset[1], position[2] + offset[2]]
                    rotate_90 = p.getQuaternionFromEuler([0, 0, -math.pi / 2])
                    planeId = p.loadURDF("custom.urdf", position_with_offset, rotate_90)
                    p.setCollisionFilterPair(robotId, planeId, -1, -1, enableCollision=0)

                    self.robotIds.append(robotId)
                    self.specimenIds.append(planeId)

                    # Initialize PID controllers for this robot
                    self.pid_controllers.append({
                        'x': PIDController(Kp=self.pid_params['x']['Kp'], Ki=self.pid_params['x']['Ki'], Kd=self.pid_params['x']['Kd'], dt=1.0 / 240),
                        'y': PIDController(Kp=self.pid_params['y']['Kp'], Ki=self.pid_params['y']['Ki'], Kd=self.pid_params['y']['Kd'], dt=1.0 / 240),
                        'z': PIDController(Kp=self.pid_params['z']['Kp'], Ki=self.pid_params['z']['Ki'], Kd=self.pid_params['z']['Kd'], dt=1.0 / 240)
                    })

                    agent_count += 1

    def set_pid_setpoints(self, robot_id, target_x, target_y, target_z):
        self.pid_controllers[robot_id]['x'].setpoint = target_x
        self.pid_controllers[robot_id]['y'].setpoint = target_y
        self.pid_controllers[robot_id]['z'].setpoint = target_z

    def run(self, actions, num_steps=1):
        for step in range(num_steps):
            for i, robot_id in enumerate(self.robotIds):
                # Get current joint positions
                current_x = p.getJointState(robot_id, 0)[0]
                current_y = p.getJointState(robot_id, 1)[0]
                current_z = p.getJointState(robot_id, 2)[0]

                # Compute PID outputs
                velocity_x = self.pid_controllers[i]['x'].compute(current_x)
                velocity_y = self.pid_controllers[i]['y'].compute(current_y)
                velocity_z = self.pid_controllers[i]['z'].compute(current_z)

                # Apply velocities
                p.setJointMotorControl2(robot_id, 0, p.VELOCITY_CONTROL, targetVelocity=velocity_x)
                p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, targetVelocity=velocity_y)
                p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=velocity_z)

            p.stepSimulation()

            if self.render:
                time.sleep(1.0 / 240)

        return self.get_states()

    def get_states(self):
        states = {}
        for robotId in self.robotIds:
            robot_position = p.getBasePositionAndOrientation(robotId)[0]
            joint_states = p.getJointStates(robotId, [0, 1, 2])

            states[f'robotId_{robotId}'] = {
                "robot_position": robot_position,
                "joint_states": joint_states
            }

        return states

    def close(self):
        p.disconnect()
