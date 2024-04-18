import numpy as np
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


@dataclass(frozen=True)
class SnakeConfig:
    num_joints: int
    density: float = 4000.0
    viscosity: float = 0.1
    gear: float = 150.0
    gravity: float = -9.80665 * 0.5

    def mujoco_xml(self) -> ElementTree.Element:
        mujoco = ElementTree.Element("mujoco")

        ElementTree.SubElement(
            mujoco,
            "compiler",
            angle="degree",
            coordinate="local",
            inertiafromgeom="true",
        )

        ElementTree.SubElement(
            mujoco,
            "option",
            collision="all",
            density=str(self.density),
            integrator="RK4",
            timestep="0.01",
            viscosity=str(self.viscosity),
            gravity=f"0 {self.gravity} 0",
        )

        default = ElementTree.SubElement(mujoco, "default")
        ElementTree.SubElement(
            default,
            "geom",
            conaffinity="1",
            condim="1",
            contype="1",
            material="geom",
            rgba="0.8 0.6 .4 1",
        )
        ElementTree.SubElement(default, "joint", armature="0.1")

        asset = ElementTree.SubElement(mujoco, "asset")
        ElementTree.SubElement(
            asset,
            "texture",
            builtin="gradient",
            height="100",
            rgb1="1 1 1",
            rgb2="0 0 0",
            type="skybox",
            width="100",
        )
        ElementTree.SubElement(
            asset,
            "texture",
            builtin="flat",
            height="1278",
            mark="cross",
            markrgb="1 1 1",
            name="texgeom",
            random="0.01",
            rgb1="0.8 0.6 0.4",
            rgb2="0.8 0.6 0.4",
            type="cube",
            width="127",
        )
        ElementTree.SubElement(
            asset,
            "texture",
            builtin="checker",
            height="100",
            name="texplane",
            rgb1="0 0 0",
            rgb2="0.8 0.8 0.8",
            type="2d",
            width="100",
        )
        ElementTree.SubElement(
            asset,
            "material",
            name="MatPlane",
            reflectance="0.5",
            shininess="1",
            specular="1",
            texrepeat="10 10",
            texture="texplane",
        )
        ElementTree.SubElement(
            asset, "material", name="geom", texture="texgeom", texuniform="true"
        )

        worldbody = ElementTree.SubElement(mujoco, "worldbody")
        ElementTree.SubElement(
            worldbody,
            "light",
            cutoff="100",
            diffuse="1 1 1",
            dir="-0 0 -1.3",
            directional="true",
            exponent="1",
            pos="0 0 1.3",
            specular=".1 .1 .1",
        )

        floor = ElementTree.SubElement(worldbody, "body", name="floor", pos="0 0 0")
        ElementTree.SubElement(
            floor,
            "geom",
            conaffinity="1",
            condim="3",
            material="MatPlane",
            name="floor_geom",
            pos="0 0 -0.1",
            rgba="0.8 0.9 0.8 1",
            size="10 10 0.1",
            type="plane",
        )
        for i in range(-10, 10 + 1):
            for j in range(-10, 10 + 1):
                ElementTree.SubElement(
                    floor,
                    "geom",
                    conaffinity="1",
                    condim="3",
                    name=f"cyl {i} {j}",
                    pos=f"{i} {j} 0.1",
                    rgba="0.0 0.0 0.8 0.8",
                    size="0.1 0.2",
                    type="cylinder",
                )

        body = ElementTree.SubElement(worldbody, "body", name="torso", pos="0 0.2 0")
        ElementTree.SubElement(
            body,
            "camera",
            name="track",
            mode="trackcom",
            fovy="59",
            pos="-2.5 0 15",
            xyaxes="1 0 0 0 1 0",
        )
        ElementTree.SubElement(
            body,
            "geom",
            name="torso_geom",
            density="1000",
            fromto="0 0 0 -0.5 0 0",
            size="0.1",
            type="capsule",
        )
        ElementTree.SubElement(
            body, "joint", axis="1 0 0", name="free_body_x", pos="0 0 0", type="slide"
        )
        ElementTree.SubElement(
            body, "joint", axis="0 1 0", name="free_body_y", pos="0 0 0", type="slide"
        )
        ElementTree.SubElement(
            body, "joint", axis="0 0 1", name="free_body_yaw", pos="0 0 0", type="hinge"
        )

        for joint_index in range(1, self.num_joints):
            body = ElementTree.SubElement(
                body, "body", name=f"mid{joint_index}", pos="-0.5 0 0"
            )
            ElementTree.SubElement(
                body,
                "geom",
                name=f"mid{joint_index}_geom",
                density="1000",
                fromto="0 0 0 -0.5 0 0",
                size="0.1",
                type="capsule",
            )
            ElementTree.SubElement(
                body,
                "joint",
                axis="0 0 1",
                limited="true",
                name=f"motor{joint_index}_rot",
                pos="0 0 0",
                range="-100 100",
                type="hinge",
            )
        body = ElementTree.SubElement(body, "body", name="back", pos="-0.5 0 0")
        ElementTree.SubElement(
            body,
            "geom",
            name="back_geom",
            density="1000",
            fromto="0 0 0 -0.5 0 0",
            size="0.1",
            type="capsule",
        )
        ElementTree.SubElement(
            body,
            "joint",
            axis="0 0 1",
            limited="true",
            name=f"motor{self.num_joints}_rot",
            pos="0 0 0",
            range="-100 100",
            type="hinge",
        )

        actuator = ElementTree.SubElement(mujoco, "actuator")
        for i in range(1, self.num_joints + 1):
            ElementTree.SubElement(
                actuator,
                "motor",
                ctrllimited="true",
                ctrlrange="-1 1",
                gear=str(self.gear),
                joint=f"motor{i}_rot",
            )
        return mujoco

    def write_xml(
        self,
        xml_path: Path,
    ):
        mujoco_xml = self.mujoco_xml()
        ElementTree.ElementTree(mujoco_xml).write(xml_path)


default_config = SnakeConfig(12)


class SnakeEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        snake_config: SnakeConfig = default_config,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-4,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.snake_config = snake_config

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(2 * (self.snake_config.num_joints + 1) + 2,),
                dtype=np.float64,
            )
        else:
            observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(2 * (self.snake_config.num_joints + 1) + 4,),
                dtype=np.float64,
            )

        model_path = Path(__file__).parent / "assets" / "snake.xml"
        self.snake_config.write_xml(model_path)

        MujocoEnv.__init__(
            self,
            "~\\projects\\cpsc533v-project\\snake_pegboard\\snake_env\\assets\\snake.xml",
            4,
            observation_space=observation_space,
            **kwargs,
        )

        self.model.opt.gravity[1] = self.snake_config.gravity

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = self._forward_reward_weight * (
            x_velocity + y_velocity
        )  # x_velocity

        ctrl_cost = self.control_cost(action)

        observation = self._get_obs()
        reward = forward_reward  # - ctrl_cost
        # reward = np.linalg.norm(xy_position_after)
        info = {
            "reward_fwd": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()

        terminated = np.max(np.abs(xy_position_after)) >= 10

        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
