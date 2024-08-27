"""
A file containing the base code required to start up spot.
"""

import argparse
from contextlib import contextmanager
import io
import logging
import time
from typing import Any, Generator, Generic, List, Literal, Tuple, TypeVar, Union
from PIL.Image import Image
import PIL.Image
from scipy.ndimage import rotate as scirotate

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from google.protobuf import duration_pb2
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.robot_id import RobotIdClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    blocking_selfright,
    blocking_sit,
    blocking_command,
    block_for_trajectory_cmd,
    block_until_arm_arrives,
)
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient
from bosdyn.client.math_helpers import Quat, SE3Pose, math
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    BODY_FRAME_NAME,
    get_a_tform_b,
)
import bosdyn.api.basic_command_pb2 as basic_command_pb2
from bosdyn.api import (
    geometry_pb2,
    trajectory_pb2,
    world_object_pb2,
    arm_surface_contact_pb2,
    arm_surface_contact_service_pb2,
    #mobility_command_pb2,
    arm_command_pb2,
    synchronized_command_pb2,
    robot_command_pb2,
    mobility_command_pb2,
)
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.util import seconds_to_timestamp
from bosdyn.client.arm_surface_contact import ArmSurfaceContactClient

import numpy as np

Num = int | float


class SpotConfig:
    """
    A configuration for a Spot robot.
    """

    def __init__(self):
        self.hostname: str = "192.168.80.3"  # The hostname of the Spot robot
        self.log_level: int = logging.ERROR  # what level to log at

    def add_args_to(self, parser: argparse.ArgumentParser) -> "SpotConfig":
        """
        Adds the arguments of the config to the given parser so that the user can query them.
        """
        # hostname
        parser.add_argument(
            "--hostname",
            default="192.168.80.3",
            help="Hostname or address of robot," ' e.g. "beta25-p" or "192.168.80.3"',
        )

        # Logging Levels
        parser.add_argument(
            "--log_level",
            help="Determines the level to log at.",
            default="warning",
            choices=["debug", "info", "warning", "error", "critical"],
        )
        return self

    def arg_parse_set(self, namespace: argparse.Namespace) -> "SpotConfig":
        """
        Sets the values of this config based on the given arg parse result.
        """
        if namespace.loglevel:
            match namespace.log_level:
                case "debug":
                    level = logging.DEBUG
                case "info":
                    level = logging.INFO
                case "error":
                    level = logging.ERROR
                case "critical":
                    level = logging.CRITICAL
                case _:
                    level = logging.WARNING

            self.log_level = level

        return self

    def arg_parse(self) -> "SpotConfig":
        """
        Parses the config arguments.
        """
        parser = argparse.ArgumentParser()
        self.add_args_to(parser)
        self.arg_parse_set(parser.parse_args())
        return self


IMAGE_PIXEL_FORMAT = Union[
    Literal["PIXEL_FORMAT_GREYSCALE_U16"],
    Literal["PIXEL_FORMAT_DEPTH_U16"],
    Literal["PIXEL_FORMAT_RGBA_U8"],
    Literal["PIXEL_FORMAT_RGB_U8"],
    Literal["PIXEL_FORMAT_GREYSCALE_U8"],
    # Literal['PIXEL_FORMAT_UNKNOWN'],
]

IMAGE_FORMAT = Union[
    Literal["FORMAT_RLE"],
    Literal["FORMAT_RAW"],
    Literal["FORMAT_JPEG"],
    # Literal['FORMAT_UNKNOWN'],
]

T = TypeVar("T")


class SpotImageResponse(Generic[T]):
    """
    A class to help return Spot's response when images are requested.
    """

    def __init__(
        self,
        frontleft: T | None = None,
        frontright: T | None = None,
        left: T | None = None,
        right: T | None = None,
        back: T | None = None,
        frontleft_response: Any | None = None,
        frontright_response: Any | None = None,
        left_response: Any | None = None,
        right_response: Any | None = None,
        back_response: Any | None = None,
    ):
        self.frontleft: T | None = frontleft
        self.frontright: T | None = frontright
        self.left: T | None = left
        self.right: T | None = right
        self.back: T | None = back

        self.frontleft_response: Any | None = frontleft_response
        self.frontright_response: Any | None = frontright_response
        self.left_response: Any | None = left_response
        self.right_response: Any | None = right_response
        self.back_response: Any | None = back_response

    def response_list(self) -> List[Any]:
        """
        Returns this response as a list of received images, omitting from the
        list the images that were not received. The images are
        otherwise in the same order as `self.maybe_response_list()` would return them.
        """
        return [v for v in self.maybe_response_list() if v is not None]

    def maybe_response_list(self) -> List[Any | None]:
        """
        Returns this response as a list of received images in
        `[frontleft, frontright, left, right, back]` order, replacing
        any images that were not requested with `None`.
        """
        return [
            self.frontleft_response,
            self.frontright_response,
            self.left_response,
            self.right_response,
            self.back_response,
        ]

    def image_list(self) -> List[T]:
        """
        Returns this response as a list of received images, omitting from the
        list the images that were not received. The images are
        otherwise in the same order as `self.maybe_image_list()` would return them.
        """
        return [v for v in self.maybe_image_list() if v is not None]

    def maybe_image_list(self) -> List[T | None]:
        """
        Returns this response as a list of received images in
        `[frontleft, frontright, left, right, back]` order, replacing
        any images that were not requested with `None`.
        """
        return [self.frontleft, self.frontright, self.left, self.right, self.back]


async def dummy():
    """
    A function is only run asynchronously after its first `await`. This function
    only exists so that said `await` can be called up front rather than later.
    """
    return None

class Fiducial:
    """
    A class that encompasses the information of a fiducial.
    """

    @staticmethod
    def from_apriltag_world_object(world_object) -> "Fiducial":
        """
        Returns a new `Fiducial` created from the given `WorldObject`.
        """
        transforms = world_object.transforms_snapshot
        apriltag_info = world_object.apriltag_properties

        tag_id = apriltag_info.tag_id
        frame_name = apriltag_info.frame_name_fiducial
        frame_name_filtered = apriltag_info.frame_name_fiducial_filtered
        #frame_name_camera = apriltag_info.frame_name_camera

        return Fiducial(transforms, tag_id, frame_name, frame_name_filtered)

    def __init__(self,
                transforms,
                tag_id: int,
                frame_name_fiducial: str,
                frame_name_fiducial_filtered: str
    ):
        # A snapshot of the frame tree.
        self.transforms_snapshot = transforms
        self.id: int = tag_id
        self.frame_name_fiducial: str = frame_name_fiducial
        self.frame_name_fiducial_filtered: str = frame_name_fiducial_filtered

    def body_t_fiducial(self) -> SE3Pose:
        """
        Returns the grav-aligned body pose relative to the fiducial pose.
        """
        return get_a_tform_b(self.transforms_snapshot, BODY_FRAME_NAME, self.frame_name_fiducial_filtered)

    def odom_t_fiducial(self) -> SE3Pose:
        """
        Returns the odom pose relative to the fiducial pose.
        """
        return get_a_tform_b(self.transforms_snapshot, ODOM_FRAME_NAME, self.frame_name_fiducial_filtered)

    def frame_t_fiducial(self, frame_name: str) -> SE3Pose | None:
        """
        Returns the pose of this `Fiducial` relative to the given frame.

        Args:
            frame_name: the name of the "frame" in `frame_t_tag`
        """
        return get_a_tform_b(self.transforms_snapshot, frame_name, self.frame_name_fiducial_filtered)

class LeaseTracker:
    """
    A small class that keeps track of how many `LeasedSpot` objects there are
    for a specific Spot, only dropping the lease when all `LeasedSpot` objects
    have stopped requesting a lease.
    """

    def __init__(self, lease_client: LeaseClient):
        self._lease_client: LeaseClient = lease_client
        self._count: int = 0
        self._lease: None | bosdyn.client.lease.LeaseKeepAlive = None

    # --- Lease Start ---

    def _start_lease(self):

        # check for previous lease
        if self._lease is not None:
            # end previous lease
            self._lease.__exit__(None, None, None)

        # start new lease
        self._lease = bosdyn.client.lease.LeaseKeepAlive(
            self._lease_client,
            must_acquire=True,
            return_at_exit=True,
        )
        self._lease.__enter__()

    def _end_lease(self, exc_type, exc_val, traceback):

        # check for lease
        if self._lease is not None:
            self._lease.__exit__(exc_type, exc_val, traceback)
        self._lease = None

    # --- Context Manager Functions ---

    def __enter__(self):
        self._count = max(self._count, 0)
        if self._count == 0:
            self._start_lease()
        self._count += 1

    def __exit__(self, exc_type, exc_val, traceback):
        self._count -= 1
        if self._count <= 0:
            self._count = 0


class SpotShared:
    """
    A class used to encompass the data shared by both leased and unleased spot robots.
    """

    def __init__(
        self,
        config: SpotConfig,
        bosdyn_sdk: bosdyn.client.Sdk,
        robot: bosdyn.client.Robot,
        image_client: ImageClient,
        power_client: PowerClient,
        lease_client: LeaseClient,
        state_client: RobotStateClient,
        command_client: RobotCommandClient,
        world_client: WorldObjectClient,
        id_client: RobotIdClient,
        manipulation_client: ManipulationApiClient,
        arm_surface_contact_client: ArmSurfaceContactClient,
    ):
        self.config: SpotConfig = config
        self.sdk: bosdyn.client.Sdk = bosdyn_sdk
        self.robot: bosdyn.client.Robot = robot
        self.image_client: ImageClient = image_client
        self.power_client: PowerClient = power_client
        self.lease_client: LeaseClient = lease_client
        self.state_client: RobotStateClient = state_client
        self.command_client: RobotCommandClient = command_client
        self.world_object_client: WorldObjectClient = world_client
        self.id_client: RobotIdClient = id_client
        self.manipulation_client: ManipulationApiClient = manipulation_client
        self.arm_surface_contact_client: ArmSurfaceContactClient = (
            arm_surface_contact_client
        )

        self.lease_tracker: LeaseTracker = LeaseTracker(lease_client)


class Spot:
    """
    A class to encapsulate the functionality of controlling a Boston Dynamics
    Spot robot.

    A Spot has a small amount of functionality that it can perform without a
    lease. If you want to do more (such as making the robot move around) you can
    use `with spot.leased() as spot: ...` to perform functionality that require
    a lease.
    """

    def __init__(
        self,
        config: SpotShared | SpotConfig | None = None,
        bosdyn_sdk: bosdyn.client.Sdk | None = None,
        setup_logging: bool = True,
    ):

        # If the config is actually an instance of `SpotShared`, then the Spot
        # is already initialized and we can just return
        if isinstance(config, SpotShared):
            self._shared = config
            return

        # --- Initalize Spot ---

        if config is None:
            config = SpotConfig()

        # The Boston Dynamics Python library uses Python's logging module to
        # generate output. Applications using the library can specify how
        # the logging information should be output.
        if setup_logging:
            bosdyn.client.util.setup_logging(config.log_level)

        # The SDK object is the primary entry point to the Boston Dynamics API.
        # create_standard_sdk will initialize an SDK object with typical default
        # parameters. The argument passed in is a string identifying the client.
        if bosdyn_sdk is None:
            sdk = bosdyn.client.create_standard_sdk("SpotClient")
        else:
            sdk = bosdyn_sdk

        # The Spot robot.
        robot = sdk.create_robot(config.hostname)
        robot.logger.setLevel(config.log_level)

        # Clients need to authenticate to a robot before being able to use it.
        bosdyn.client.util.authenticate(robot)

        robot.sync_with_directory()

        # Establish time sync with the robot. This kicks off a background thread to establish time sync.
        # Time sync is required to issue commands to the robot. After starting time sync thread, block
        # until sync is established.
        robot.time_sync.wait_for_sync()

        # The robot state client will allow us to get the robot's state information, and construct
        # a command using frame information published by the robot.
        robot.logger.info("Ensuring State Client")
        state_client = robot.ensure_client(RobotStateClient.default_service_name)

        robot.logger.info("Ensuring Image Client")
        image_client = robot.ensure_client(ImageClient.default_service_name)

        robot.logger.info("Ensuring Command Client")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        robot.logger.info("Ensuring Power Client")
        power_client = robot.ensure_client(PowerClient.default_service_name)

        # The client
        robot.logger.info("Ensuring World Object Client")
        world_object_client = robot.ensure_client(
            WorldObjectClient.default_service_name
        )

        robot.logger.info("Ensuring Robot ID Client")
        id_client = robot.ensure_client(RobotIdClient.default_service_name)

        robot.logger.info("Ensuring Robot Manipulation Client")
        manip_client = robot.ensure_client(ManipulationApiClient.default_service_name)

        robot.logger.info("Ensuring Robot Arm Surface Contact Client")
        arm_surface_contact_client = robot.ensure_client(
            ArmSurfaceContactClient.default_service_name
        )

        # Only one client at a time can operate a robot. Clients acquire a lease to
        # indicate that they want to control a robot. Acquiring may fail if another
        # client is currently controlling the robot. When the client is done
        # controlling the robot, it should return the lease so other clients can
        # control it. The LeaseKeepAlive object takes care of acquiring and returning
        # the lease for us.
        robot.logger.info("Ensuring Robot Lease Client")
        lease_client = robot.ensure_client(LeaseClient.default_service_name)

        self._shared: SpotShared = SpotShared(
            config,
            sdk,
            robot,
            image_client,
            power_client,
            lease_client,
            state_client,
            command_client,
            world_object_client,
            id_client,
            manip_client,
            arm_surface_contact_client,
        )

    def image_client(self) -> ImageClient:
        """Returns Spot's image client."""
        return self._shared.image_client

    def power_client(self) -> PowerClient:
        """Returns Spot's power client."""
        return self._shared.power_client

    def lease_client(self) -> LeaseClient:
        """Returns Spot's lease client."""
        return self._shared.lease_client

    def state_client(self) -> RobotStateClient:
        """Returns Spot's state client."""
        return self._shared.state_client

    def command_client(self) -> RobotCommandClient:
        """Returns Spot's command client."""
        return self._shared.command_client

    def world_object_client(self) -> WorldObjectClient:
        """Returns Spot's world object client."""
        return self._shared.world_object_client

    def id_client(self) -> RobotIdClient:
        """Returns Spot's id client."""
        return self._shared.id_client

    def manipulation_api_client(self) -> ManipulationApiClient:
        """Returns Spot's manipulation api client."""
        return self._shared.manipulation_client

    def arm_surface_contact_client(self) -> ArmSurfaceContactClient:
        """Returns Spot's surface contact client."""
        return self._shared.arm_surface_contact_client

    async def robot_state(self):
        """Returns the robot state information."""
        return self.state_client().get_robot_state()

    async def robot_hardware(self):
        """Returns the robot hardware information."""
        return self.state_client().get_hardware_config_with_link_info()

    async def robot_metrics(self):
        """Returns the robot's metrics."""
        return self.state_client().get_robot_metrics()

    def has_arm(self, timeout: Num | None = None) -> bool:
        """
        Returns `true` if the robot has an arm and `false` otherwise.

        Args:
           timeout: If the robot does not respond in `timeout` seconds then an error will be thrown. If the value is `None` then the command will never time out.
        """
        return self.robot().has_arm(timeout)

    @contextmanager
    def leased(
        self, check_estop: bool = True, take: bool = False
    ) -> Generator["LeasedSpot", Any, Any]:
        """
        Returns a context manager that yields a leased handle on Spot. The
        leased handle allows you to do more advanced things like move the
        Spot around or power it on/off.

        Args:
            check_estop: whether to check for whether Spot is currently estopped or not
            take: Will forcefully take the lease if necessary. Useful for taking the lease from the tablet so that the robot can be positioned via the tablet before it passes the lease to us and we run our code.
        """

        if take:
            self.lease_client().take()

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        if check_estop:

            # check if there is currently an estop
            # if not robot.is_estopped() and create_estop_if_none:
            # no estop currently so start up a new process that will display the
            # estop as a GUI
            # os.system(f'start cmd /k ; python ./estop_gui.py {config.hostname}')
            # with open(os.devnull, 'r+b', 0) as devnull:
            #    Popen(['python', './estop_gui.py', f'{config.hostname}'],
            #            stdin=devnull, stdout=STDOUT, stderr=STDOUT, close_fds=True)
            # time.sleep(3) # give the estop process a few seconds to launch

            # assert that we have an estop
            assert not self.robot().is_estopped(), (
                "Robot is estopped. Please use an external E-Stop client, "
                "such as the estop SDK example, to configure E-Stop."
            )

        with self._shared.lease_tracker as _:
            yield LeasedSpot(self._shared)

    def config(self) -> SpotConfig:
        """
        Returns Spot's configuration.
        """
        return self._shared.config

    def robot(self) -> bosdyn.client.Robot:
        """
        Returns the raw Spot robot client from boston dynamics.
        """
        return self._shared.robot

    async def fiducials_in_view(self) -> list[Fiducial]:
        """
        Returns a list of all fiducials currently in view.
        """
        await dummy()
        return [
            Fiducial.from_apriltag_world_object(obj)
            for obj in self.world_object_client()
            .list_world_objects(object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG])
            .world_objects
        ]

    async def transforms_snapshot(self, robot_state = None):
        """
        Returns the transforms snapshot from the given robot_state or from a newly-gotten state if the given one is `None`.
        """
        if robot_state is None:
            robot_state = await self.robot_state()
        return robot_state.kinematic_state.transforms_snapshot

    NAMED_ROTATION = Literal["floor", "ceil", "forward", "backward", "left", "right"]

    async def named_rotation(self, name: NAMED_ROTATION, frame_name: str, robot_state_transforms_snapshot = None) -> Quat:
        """
        Returns the quaternion in the given frame that corresponds to the given named rotation.

        Args:
            name: the named rotation
            frame_name: the frame that the rotation should be in
            robot_state_transforms_snapshot: The transforms from `spot.transforms_snapshot()` so that the given named rotation can be converted to the desired rotation. If `None`, then the transforms are gotten by the function.
        """

        if robot_state_transforms_snapshot is None:
            robot_state_transforms_snapshot = await self.transforms_snapshot()

        # get the rotation in the GRAV_ALIGNED_BODY_FRAME_NAME frame
        if name == "floor":
            rot = Quat.from_pitch(math.pi / 2)
        elif name == "ceil":
            rot = Quat.from_pitch(-math.pi / 2)
        elif name == "forward":
            rot = Quat.from_pitch(0)
        elif name == "backward":
            rot = Quat.from_pitch(math.pi)
        elif name == "left":
            rot = Quat.from_yaw(math.pi / 2)
        elif name == "right":
            rot = Quat.from_yaw(-math.pi / 2)
        else:
            raise AssertionError(f"{rot} was not a known named rotation")

        flatbody_t_frame: Quat = get_a_tform_b(robot_state_transforms_snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, frame_name).rotation
        return (flatbody_t_frame * rot)

    BODYCAM_NAME_TYPE = Literal["frontleft", "frontright", "left", "right", "back"]
    BODYCAM_NAME: BODYCAM_NAME_TYPE = [
        "frontleft",
        "frontright",
        "left",
        "right",
        "back",
    ]

    BODYCAM_ROTATION = {
        "frontleft": -90,
        "frontright": -90,
        "left": 0,
        "right": 180,
        "back": 0,
    }

    async def get_images(
        self,
        frontleft: bool = False,
        frontright: bool = False,
        left: bool = False,
        right: bool = False,
        back: bool = False,
        rgb: bool = False,
        greyscale: bool = False,
        depth: bool = False,
        rotate: bool = False,
        quality_percent: int = 100,
        depth_in_visual_frame: bool | None = None,
    ) -> Tuple[
        SpotImageResponse[Image],
        SpotImageResponse[Image],
        SpotImageResponse[np.ndarray[np.uint16]],
        list[Any],
    ]:
        """
        Returns a list of images from spot.

        Args:
            frontleft: whether to request an image from the front-left camera
            frontright: whether to request an image from the front-right camera
            left: whether to request an image from the left camera
            right: whether to request an image from the right camera
            back: whether to request an image from the back camera

            rgb: whether an RGB image should be gotten from each of the requested cameras
            greyscale: whether a greyscale image should be gotten from each of the requested cameras
            depth: whether a depth image should be gotten from each of the requested cameras

            rotate: whether to rotate the images so that they are all upright (otherwise some of them are on their side or upside down)

            quality_percent: a value in the range [0, 100] used to determine how high-res the images should be (0 is 0% resolution and 100 is %100 resolution, with values in-between being some percantage between %0 and %100 possible resolution)

            depth_in_visual_frame: If True, the depth image will have its positions aligned with the positions in the greyscale and rgb images taken in the same call. If False, then the greyscale and rgb images taken in the same call will be aligned with the depth images. If None (the default), all images will be taken independantly.

        Returns:
            a tuple of `(rgb, greyscale, depth)` image responses that contain the requested images or `None` for images that were not requested
        """
        await dummy()  # the rest of this function is done async

        requests = [
            r if c else None
            for r, c in zip(
                Spot.BODYCAM_NAME, [frontleft, frontright, left, right, back]
            )
        ]

        # --- Create the requests ---

        reqs: List[None | str | Tuple[str, str, str, str]] = []

        # req rgb
        for req in requests:
            if not rgb or req is None:
                reqs.append(None)
                continue
            cam = (
                f"{req}_fisheye_image"
                if depth_in_visual_frame is not False
                else f"{req}_visual_in_depth"
            )
            reqs.append((cam, req, "rgb", "PIXEL_FORMAT_RGB_U8"))

        # req greyscale
        for req in requests:
            if not greyscale or req is None:
                reqs.append(None)
                continue
            cam = (
                f"{req}_fisheye_image"
                if depth_in_visual_frame is not False
                else f"{req}_visual_in_depth"
            )
            reqs.append((cam, req, "greyscale", "PIXEL_FORMAT_GREYSCALE_U8"))

        # req depth
        for req in requests:
            if not depth or req is None:
                reqs.append(None)
                continue
            cam = (
                f"{req}_depth_in_visual_frame"
                if depth_in_visual_frame is True
                else f"{req}_depth"
            )
            reqs.append(cam)

        # --- Send the requests and receive the images ---

        q = min(max(quality_percent, 0), 100)

        built_reqs = []
        for req in reqs:
            if req is None:
                continue
            elif isinstance(req, str):
                built_reqs.append(build_image_request(req, quality_percent=q))
            else:
                built_reqs.append(
                    build_image_request(
                        req[0],
                        image_format="FORMAT_JPEG",
                        pixel_format=req[3],
                        quality_percent=q,
                    )
                )

        image_responses = self.image_client().get_image(built_reqs)

        # --- Handle the received images ---

        # create the output tuple
        i = 0

        images = []
        responses = []

        for req in reqs:

            if req is None:
                images.append(None)
                responses.append(None)
                continue

            response = image_responses[i]
            i += 1

            (rows, cols) = (response.shot.image.rows, response.shot.image.cols)
            # pixel_format = image.shot.image.pixel_format
            data = response.shot.image.data

            if isinstance(req, str):
                # depth is raw bytestream
                img = np.frombuffer(data, dtype=np.uint16)
                img = img.reshape(rows, cols)
                if rotate:
                    img = scirotate(img, angle=self.BODYCAM_ROTATION[loc])
                images.append(img)
                responses.append(response)

            else:
                (_, loc, ty, _) = req

                if ty == "rgb":
                    # rgb images are JPEG
                    img = PIL.Image.open(io.BytesIO(data))
                    if rotate:
                        img = img.rotate(self.BODYCAM_ROTATION[loc], expand=True)
                    images.append(img)
                    responses.append(response)

                elif ty == "greyscale":
                    # greyscale images are JPEG
                    img = PIL.Image.open(io.BytesIO(data))
                    if rotate:
                        img = img.rotate(self.BODYCAM_ROTATION[loc], expand=True)
                    images.append(img)
                    responses.append(response)

        return (
            SpotImageResponse(*images[0:5], *responses[0:5]),
            SpotImageResponse(*images[5:10], *responses[5:10]),
            SpotImageResponse(*images[10:15], *responses[10:15]),
        )

    async def are_motors_powered_on(self, timeout: Num | None = None) -> bool:
        """
        Returns `True` if the robot is currently powered on and returns `False` otherwise.

        Args:
            cut_immediately: If `True`, then the robot will immediately have its power cut off, making it fall to the ground if it is standing. Otherwise, the robot sits down gently and then powers off.
            timeout_sec: the max number of seconds to wait for the command to complete

        Throws: error if `timeout_sec` seconds pass but the command has yet to complete
        """
        await dummy()
        return self.robot().is_powered_on(timeout)


class LeasedSpot(Spot):
    """
    A class to encapsulate the functionality of controlling a Boston Dynamics
    Spot robot once it has been leased.
    """

    def __init__(self, shared: SpotShared):
        super().__init__(shared, shared.sdk, False)
        self._shared: SpotShared = shared

    async def stop(self):
        """
        Makes Spot stop doing everything that it is doing at the moment with
        minimal movement.
        """
        await dummy()
        self.command_client().robot_command(RobotCommandBuilder.stop_command())

    async def power_motors_on(self, timeout: Num = 20) -> bool:
        """
        Powers on the Spot if it is not powered on already.

        Args:
            timeout_sec: the number of seconds Spot has to power on before an error is thrown
        """
        await dummy()

        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        self.robot().logger.info("Powering on robot... This may take several seconds.")
        self.robot().power_on(timeout_sec=float(timeout))
        assert await self.are_motors_powered_on(), "Robot power on failed."
        self.robot().logger.info("Robot powered on.")

    async def power_motors_off(
        self, cut_immediately: bool = False, timeout: Num = 20
    ) -> bool:
        """
        Powers off Spot's motors.

        Args:
            cut_immediately: If `True`, then the robot will immediately have its power cut off, making it fall to the ground if it is not already sitting. Otherwise, the robot sits down gently and then powers its motors off.
            timeout_sec: the max number of seconds to wait for the command to complete

        Throws:
            an error if `timeout_sec` seconds pass but the command has yet to complete

        Returns `True` the power was successufully turned off and `False` otherwise.
        """
        await dummy()

        if await self.are_motors_powered_on(timeout):
            self.robot().logger.info("Robot Powering off...", cut_immediately)
            # Power the robot off. By specifying "cut_immediately=False", a safe power off command
            # is issued to the robot. This will attempt to sit the robot before powering off.
            self.robot().power_off(
                cut_immediately=cut_immediately, timeout_sec=float(timeout)
            )
            assert not self.are_motors_powered_on(), "Robot power off failed."
            self.robot().logger.info(
                "Robot powered off (cut_immediately=%s).", cut_immediately
            )
            return True

        return False

    async def stand(self, timeout: Num = 10):
        """
        Make the robot stand up.

        Args:
            height: the height that Spot should stand relative to its normal height
            timeout_sec: the max number of seconds to wait for the command to complete

        Throws: error if `timeout_sec` seconds pass but the command has yet to complete
        """
        await dummy()

        def check_stand_status(response):
            status = response.feedback.synchronized_feedback.mobility_command_feedback.stand_feedback.status
            return status == basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING

        blocking_command(
            self.command_client(),
            RobotCommandBuilder.synchro_stand_command(),
            check_stand_status,
            timeout_sec=timeout,
        )

    async def sit(self, timeout: Num = 10):
        """
        Make the robot sit down.

        Args:
            timeout_sec: the max number of seconds to wait for the command to complete

        Throws: error if `timeout_sec` seconds pass but the command has yet to complete
        """
        await dummy()
        blocking_sit(self.command_client(), timeout_sec=float(timeout))

    async def selfright(self, timeout: Num = 10):
        """
        Make the robot right itself (assuming it fell over or is on its back).

        Args:
            timeout_sec: the max number of seconds to wait for the command to complete

        Throws: error if `timeout_sec` seconds pass but the command has yet to complete
        """
        await dummy()
        blocking_selfright(self.command_client(), timeout_sec=float(timeout))

    async def battery_change_pose(self, roll_right: bool):
        """
        Make the robot sit down (if not already sitting) and roll into its side for easier battery access.

        Args:
            roll_right: if `True` then spot rolls to the right, otherwise Spot rolls to the left.
        """
        await dummy()
        self.command_client().robot_command(
            RobotCommandBuilder.battery_change_pose_command(
                dir_hint=1 if roll_right else 2
            )
        )

    async def stow_arm(self, timeout: Num | None = None, extra_safe: bool = False):
        """
        Stows Spot's arm.

        Args:
            timeout: the max amount of time (in seconds) Spot has to complete the command before throwing an error
            extra_safe: Some commands (such as the `arm_floor_contact` command) leave the robot in a state where stowing the arm causes it to sometimes slam into the robot itself. Passing in `True` for this value will cause a command to be issued prior to stowing the arm that will leave it in a state where it can safely stow the arm without slaming it down.

        Exceptions:
            AssertionError: if Spot is not outfitted with an arm
        """
        assert self.has_arm(), "Spot requires an arm to stow its arm"
        await dummy()

        if extra_safe:
            # reset the arm position to make sure that it never slams when stowing the arm
            await self.set_ee_pose(x=0.8, z=0.5, rot="forward", move_time=2)

        stow_id = self.command_client().robot_command(
            RobotCommandBuilder.arm_stow_command()
        )
        block_until_arm_arrives(self.command_client(), stow_id, timeout)

    async def unstow_arm(self, timeout: Num | None = None):
        """
        Unstows Spot's arm.

        Args:
            timeout: the max amount of time (in seconds) Spot has to complete the command before throwing an error

        Exceptions:
            AssertionError: if the command stalls or is cancelled before the arm can be complete being unstowed
        """
        assert self.has_arm(), "Spot requires an arm to unstow its arm"
        await dummy()
        unstow_id = self.command_client().robot_command(
            RobotCommandBuilder.arm_ready_command()
        )
        block_until_arm_arrives(self.command_client(), unstow_id, timeout)

    # --- Robot Movement ---

    def _robot_state(self):
        """Returns Spot's raw state."""
        return self.state_client().get_robot_state()

    async def move_and_rotate(self, vx: Num, vy: Num, vrot: Num, duration: Num):
        """
        Makes Spot move and rotate at the same time.

        Args:
            vx: the velocity (in m/s) to move in the x direction
            vy: the velocity (in m/s) to move in the y direction
            vrot: the velocity to rotate with
            duration: how long to move and rotate for
        """
        await dummy()

        # duration must be a minimum number of seconds long or else Spot ignores it
        duration = max(duration, 0.2)

        cmd_id = self.command_client().robot_command(
            RobotCommandBuilder.synchro_velocity_command(
                v_x=vx, v_y=vy, v_rot=vrot
            ),
            end_time_secs=time.time() + max(duration, 0),
        )

        # block until the command completes

        timeout_sec = duration

        if timeout_sec is not None:
            start_time = time.time()
            end_time = start_time + timeout_sec
            now = time.time()

        while timeout_sec is None or now < end_time:
            feedback_resp = self.command_client().robot_command_feedback(cmd_id)
            feedback = feedback_resp.feedback.synchronized_feedback.mobility_command_feedback

            if feedback.status != basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING:
                break

            time.sleep(0.02)
            now = time.time()

    async def move(self, vx: Num, vy: Num, duration: Num):
        """
        Makes the Spot move with the given x and y velocity for the given number of seconds.

        Args:
            vx: the velocity (in m/s) for Spot to move in for the x direction
            vy: the velocity (in m/s) for Spot to move in for the y direction
            duration: how long for Spot to move for (in seconds)
        """
        await self.move_and_rotate(vx, vy, 0, duration)

    async def rotate(self, vrot: Num, duration: Num):
        """
        Makes Spot rotate at the given velocity for the given number of seconds.
        """
        await self.move_and_rotate(0, 0, vrot, duration)

    async def move_forward(self, velocity: Num, duration: Num = 1):
        """
        Moves the robot fortward the given number of meters/second for the given
        number of seconds.

        Args:
            velocity: the velocity to move at in meters per second
            duration: how long for Spot to move for (in seconds)
        """
        await self.move(velocity, 0, duration)

    async def move_backward(self, velocity: Num, duration: Num = 1):
        """
        Moves the robot backward the given number of meters/second for the given
        number of seconds.

        Args:
            velocity: the velocity to move at in meters per second
            duration: the number of seconds to rotate for
        """
        await self.move(-velocity, 0, duration)

    async def move_left(self, velocity: Num, duration: Num = 1):
        """
        Moves the robot left the given number of meters/second for the given
        number of seconds.

        Args:
            velocity: the velocity to move at in meters per second
            duration: the number of seconds to rotate for
        """
        await self.move(0, velocity, duration)

    async def move_right(self, velocity: Num, duration: Num = 1):
        """
        Moves the robot right the given number of meters/second for the given
        number of seconds.

        Args:
            velocity: the velocity to move at in meters per second
            duration: the number of seconds to rotate for
        """
        await self.move(0, -velocity, duration)

    async def rotate_left(self, velocity: Num, duration: Num = 1):
        """
        Rotates the robot left the given number of radians/second for the given
        number of seconds.

        Args:
            velocity: the velocity to rotate by in radians per second
            duration: the number of seconds to rotate for
        """
        await self.rotate(velocity, duration)

    async def rotate_right(self, velocity: Num, duration: Num = 1):
        """
        Rotates the robot right the given number of radians/second for the given
        number of seconds.

        Args:
            velocity: the velocity to rotate by in radians per second
            duration: the number of seconds to rotate for
        """
        await self.rotate(-velocity, duration)

    async def move_to(self, x: Num, y: Num, angle: Num, frame_name: str = GRAV_ALIGNED_BODY_FRAME_NAME, max_x_velocity_limit: Num = 1, max_y_velocity_limit: Num = 1, max_rotational_velocity: Num = 1, travel_time: Num = 10, timeout: Num | None = None):
        """
        Assuming that Spot is on a 2D surface, moves to the given (x, y)
        position such that the body has the given angle at the given position.

        Args:
            x: the x of the position to move to
            y: the y of the position to move to
            angle: the angle the robot should be at at the given (x, y) position
            frame_name: what frame the (x, y) position and angle are in
            max_linear_velocity: the maximum velocity the robot is allowed to have while traveling to the given (x, y) position
            max_rotational_velocity: the maximum velocity the robot is allowed to have while rotating to get to the given desired angle of rotation
            travel_time: How many seconds Spot should take to reach the goal position/rotation. This determines its positional and angular velocity.
            timeout: how many seconds Spot has to reach the goal
        """
        await dummy()

        travel_time = max(0, travel_time)
        if timeout is None:
            timeout = travel_time * 2
        timeout = max(0, timeout)

        params = RobotCommandBuilder.mobility_params()
        params.vel_limit.CopyFrom(geometry_pb2.SE2VelocityLimit(
            max_vel=geometry_pb2.SE2Velocity(
                linear=geometry_pb2.Vec2(
                    x=max_x_velocity_limit,
                    y=max_y_velocity_limit
                ),
                angular=max_rotational_velocity
            )
        ))

        # Limit the speed so we don't charge at the person.
        se2_pose = geometry_pb2.SE2Pose(position=geometry_pb2.Vec2(x=x, y=y), angle=angle)
        move_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
            se2_pose,
            frame_name=frame_name,
            params=params
        )
        cmd_id = self.command_client().robot_command(
            command=move_cmd,
            end_time_secs=time.time() + time
        )

        # Wait until the robot repots that it is at the goal.
        block_for_trajectory_cmd(self.command_client(), cmd_id, timeout_sec=timeout)

    # --- Spot Arm Movement ---

    async def set_gripper_open_fraction(
        self, percent: Num, timeout: Num | None = None
    ):
        """
        Sets how open the Spot arm's gripper is.

        Args:
            percent: 0 is not open at all and 1 is fully open, with values in between acting as percentages of how open the gripper should be
        """
        assert self.has_arm(), "Spot requires an arm to open its gripper"
        await dummy()
        stow_id = self.command_client().robot_command(
            RobotCommandBuilder.claw_gripper_open_fraction_command(percent)
        )
        block_until_arm_arrives(self.command_client(), stow_id, timeout)

    async def set_ee_pose(
        self,
        x: Num = 0,
        y: Num = 0,
        z: Num = 0,
        rot: Quat | Spot.NAMED_ROTATION = "floor",
        frame_name: str = GRAV_ALIGNED_BODY_FRAME_NAME,
        move_time: Num = 5,
        body_may_move: bool = False,
        robot_state_transforms_snapshot = None,
    ):
        """
        Sets the pose of the end effector (sets the end-effector's position
        and rotation in euclidean space relative to the robot's current
        position).

        The position is in meters in the given frame.

        Args:
            x: the x of the position to move to (+x is forward in the body frame)
            y: the y of the position to move to (+y is to Spot's right in the body frame)
            z: the z of the position to move to (+z is up in the body frame)
            rot: The rotation that the gripper should have at its destination. If the rotation is a string, then it must be one of the string literals specified. The string literals are specified based on the "GRAV_ALIGNED_BODY_FRAME" and may be inaccurately named for other frames.
            frame_name: what frame the position/rotation is relative to
            move_time: How long (in seconds) the move to the new pose. If the given time is not enough to make the move safely then the move will instead move as fast as possible to the location (with reasonable defaults).
            body_may_move: `False` if the body cannot be moved and `True` if the body can move to allow the end-effector to take on the given pose even for poses that would otherwise be out of reach
        """
        await dummy()

        if isinstance(rot, str):
            rot = await self.named_rotation(rot, frame_name, robot_state_transforms_snapshot)

        # build the command
        arm_command = RobotCommandBuilder.arm_pose_command(
            x, y, z, rot.w, rot.x, rot.y, rot.z, frame_name, move_time
        )

        # if the body can move then we have to modify the command to tell Spot that the body can move
        if body_may_move:
            follow_arm_command = RobotCommandBuilder.follow_arm_command()
            command = RobotCommandBuilder.build_synchro_command(arm_command, follow_arm_command)
        else:
            command = RobotCommandBuilder.build_synchro_command(arm_command)

        # send the command
        cmd_id = self.command_client().robot_command(command)

        # Wait for the move to complete
        block_until_arm_arrives(self.command_client(), cmd_id)

    async def arm_follow_trajectory(
        self,
        poses: list[tuple[Num, Num, Num] | tuple[Num, Num, Num, Quat]],
        frame_name: str = GRAV_ALIGNED_BODY_FRAME_NAME,
        rotation: Quat | Spot.NAMED_ROTATION = "floor",
        gripper_open_percent: Num = 0,
        velocity: Num = 0.1,
        body_may_move: bool = False,
    ):
        """
        Makes Spot's arm follow the given trajectory of poses.

        Args:
            poses: the poses for the arm to follow
            frame_name: the name of the frame that the poses are in
            rotation: the default rotation that each pose missing a rotation will be given as its rotation
            gripper_open_percent: how open the gripper should bo
            velocity: the speed at which the poses should be followed
            body_may_move: `True` if the body can move to follow the trajectory and `False` if the arm should simply do its best to follow the trajectory without moving its body
        """
        if len(poses) == 0:
            return

        if isinstance(rotation, str):
            rotation = await self.named_rotation(rotation, frame_name)

        nposes: list[SE3Pose] = []
        for pose in poses:
            if len(pose) == 4:
                (x, y, z, r) = pose
                nposes.append(SE3Pose(x, y, z, r))
            else:
                (x, y, z) = pose
                nposes.append(SE3Pose(x, y, z, rotation))

        (_, traj) = create_arm_trajoctory(nposes, velocity)

        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            pose_trajectory_in_task=traj,
            root_frame_name=frame_name,
        )

        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_cartesian_command=arm_cartesian_command
        )

        if body_may_move:
            synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
                arm_command=arm_command,
                mobility_command=mobility_command_pb2.MobilityCommand.Request(
                    follow_arm_request=basic_command_pb2.FollowArmCommand.Request()
                )
            )
        else:
            synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
                arm_command=arm_command,
            )

        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            gripper_open_percent,
            build_on_command=robot_command
        )

        cmd_id = self.command_client().robot_command(robot_command)

        block_until_arm_arrives(self.command_client(), cmd_id)

    async def arm_floor_contact(
        self,
        points: list[tuple[Num, Num]],
        frame_name: str = GRAV_ALIGNED_BODY_FRAME_NAME,
        press: Num = 0.01,
        body_can_move: bool = False,
        velocity: Num = 0.1,
        gripper_open_percent: Num = 0,
        await_floor_touch: bool = True,
    ):
        """
        Moves the arm of the robot across the ground in accordance with the given points.

        The pressure is constant so a negative pressure will constantly move the
        arm away from the ground and a positive pressure will constantly move it
        towards the ground. This means that if it does not start off touching
        the ground then the arm will appear to "drift" over time because the
        force will continue to be applied and will continue to move down (or up)
        until something gets in its way.

        WARNING: You can knock the robot over with this so keep the pressure low.
        0.05 is typically enough to press but not knock the robot over.

        Args:
            points: the list of (x, y) points, with positive x pointing in front of the robot and positive y moving to the right of the robot
            press: how much pressure to apply to the ground. -1 is full speed away from the ground, 0 is no move, and 1 is full press into the ground. This is a range so any value in between these values is a pressure percentage between -1 (for full force away from the ground) and 1 (for pressing as much into the ground as possible).
            frame_name: the name of the frame that the points are in
            body_can_move: If `True` then the body may follow the hand to allow the hand to get to some of the given (x, y) coordinates, otherwise the body must stay still and will do its best to get to every position (only reaching as far as possible towards the position if it cannot reach it).
            velocity: how fast (in m/s) to draw the given points on the ground
            gripper_open_percent: how open the gripper should be, with 0 being 0% open and 1 being %100 open
            await_floor_touch: If `True`, then this command will wait a few seconds for the end effector to touch the floor before it begins executing the command. Otherwise, it goes straight to executing the command regardless of whether the hand has touched the ground.
        """
        await dummy()

        # bound the press force
        press = min(max(press, -1), 1)

        # if no points, then nothing to do
        if len(points) == 0:
            return
        if len(points) == 1:
            points = [points[0]] * 100

        # need at least 100 points for expected behavior
        while len(points) < 100:
            npoints = [points[0]]
            for ( x,  y) in points[1:]:
                (px, py) = npoints[-1]
                npoints.append((lerp(px, x, 0.5), lerp(py, y, 0.5)))
                npoints.append((x, y))
            points = npoints

        # get frame to convert the given GRAV_ALIGNED_BODY_FRAME points into ODOM_FRAME points
        odom_t_flat_body = get_a_tform_b(
            await self.transforms_snapshot(),
            ODOM_FRAME_NAME,
            frame_name
        )

        # create the rotation of the arm at every point
        rot = Quat.from_pitch(math.pi / 2) # point hand straight down

        # generate the poses for the points
        poses: list[SE3Pose] = [] # list of SE3Pose PROTOS (not math_helpers.SE3Pose)
        for (x, y) in points:

            # convert point to pose
            body_t_hand = SE3Pose(x=x, y=y, z=0, rot=rot) # z is ignored because we have a force applied to the ground in that direction instead

            # convert pose in flat body frame to odom frame
            odom_t_hand = odom_t_flat_body * body_t_hand

            # append the new pose to the list
            poses.append(odom_t_hand)

        # move the arm to a position in front of the robot and facing down at the floor
        #await self.set_ee_pose(points[0][0], points[0][1], 0, "floor", frame_name=GRAV_ALIGNED_BODY_FRAME_NAME, move_time=1)

        def inner(poses: list[SE3Pose]) -> Num:
            # create a trajectory through all the given poses
            traj_time, traj = create_arm_trajoctory(poses, velocity)

            gripper_cmd_packed = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper_open_percent)
            gripper_command = gripper_cmd_packed.synchronized_command.gripper_command.claw_gripper_command

            cmd = arm_surface_contact_pb2.ArmSurfaceContact.Request(
                pose_trajectory_in_task=traj,
                root_frame_name=ODOM_FRAME_NAME,
                # the API flips the press so that negative is more press
                press_force_percentage=geometry_pb2.Vec3(x=float(0), y=float(0), z=float(-press)),
                x_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_POSITION,
                y_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_POSITION,
                z_axis=arm_surface_contact_pb2.ArmSurfaceContact.Request.AXIS_MODE_FORCE,
                z_admittance=arm_surface_contact_pb2.ArmSurfaceContact.Request.ADMITTANCE_SETTING_LOOSE,
                # Enable the cross term so that if the arm gets stuck in a rut, it will retract
                # upwards slightly, preventing excessive lateral forces.
                xy_to_z_cross_term_admittance=arm_surface_contact_pb2.ArmSurfaceContact.Request.ADMITTANCE_SETTING_VERY_STIFF,
                #max_linear_velocity=int(max(velocity, 0.0001)),
                #max_linear_velocity=float(velocity), # this makes it error for some reason
                gripper_command=gripper_command
            )

            # Enable/Disable walking.
            cmd.is_robot_following_hand = body_can_move

            # A bias force (in this case, leaning forward) can help improve stability.
            cmd.bias_force_ewrt_body.CopyFrom(geometry_pb2.Vec3(x=-25, y=0, z=0))

            # send the command and wait for it to finish
            self.arm_surface_contact_client().arm_surface_contact_command(
                arm_surface_contact_service_pb2.ArmSurfaceContactCommand(request=cmd)
            )

            return traj_time

        if await_floor_touch:
            inner([poses[0]]) # move to first point
            time.sleep(4) # wait for ee to touch the ground

        # now do the rest of the poses
        traj_time = inner(poses)
        time.sleep(traj_time + 1) # wait for trajectory to finish

    async def set_arm_joint_angles(
        self,
        angles: (
            tuple[Num, Num, Num, Num, Num, Num]
            | list[tuple[Num, Num, Num, Num, Num, Num]]
        ) = (0, 0, 0, 0, 0, 0),
        times: Num | list[Num] = 1,
        max_acceleration: Num = 1,
        max_velocity: Num = 1,
        timeout: Num | None = None,
    ):
        """
        Sets the joint angles of the arm.

        Spot's arm has 6 joints. The angles are ordered from the joint at the
        base of the arm at index 0 to the joint at the gripper at index 5.
        Use the `set_gripper_open_fraction` function if you want to set how
        open/closed the gripper of the arm is.

        Args:
            angles: the angle(s) that the joints should be set to
            times: For a given index `i`, `times[i]` describe how long Spot has to change its arm's angles from those stored in `angles[i - 1]` to the angles in `angles[i]`. `times[0]` describes how long Spot has to move its arm from its current position to the one described by angles `angles[0]`. If `times` is a number, then that number is used for all times i.e. it becomes `[times] * len(angles)`.
            max_acceleration: the maximum acceleration that Spot's arm is allowed to undergo while moving its joints
            max_velocity: the maximum velocity that Spot's arm is allowed to travel at
            timeout: How long (in seconds) Spot has to move its arm through all the angles in `angles` before timing out. Note that if the times in `times` add up to an amount greater than this value then this method is gaurenteed to always time out.
        """
        await dummy()

        # if given a tuple, turn it into a list of length 1
        if isinstance(angles, tuple):
            angles = [angles]

        # times must be an array with the same length as `angles`
        if isinstance(times, (int, float)):
            times = [times]
        if len(times) == 0:
            times.append(1)
        times.extend([times[-1]] * max(0, len(times) - len(angles)))

        # move the arm through the angles
        while len(angles) != 0:

            # we can only send a max of 250 points at once to Spot
            cangles = angles[:250]
            angles  = angles[249:] # 249 so that we start at the last set of angles of previous request

            # convert ctimes to be offsets from 0
            ctimes = [
                (times[i - 1] + times[i] if (i - 1) > 0 else times[i])
                for i in range(0, len(cangles))
            ]
            times = times[249:]

            # the `time`s will be relative to the current time
            ref_time = seconds_to_timestamp(time.time())

            # create the command
            robot_cmd = RobotCommandBuilder.arm_joint_move_helper(
                joint_positions=cangles,
                times=ctimes,
                ref_time=ref_time,
                max_acc=max_acceleration,
                max_vel=max_velocity,
            )

            # perform the command, blocking until it is done
            cmd_id = self.command_client().robot_command(robot_cmd)
            block_until_arm_arrives(self.command_client(), cmd_id, timeout)

def lerp(v0: Num, v1: Num, t: Num) -> float:
    """Linearly interpolates between `v0` and `v1` based on `t` (where `t` is a value in the mathmatical range `[0, 1]`)."""
    return ((1 - float(t)) * float(v0)) + (float(t) * float(v1))

def create_arm_trajoctory(
    poses: list[SE3Pose],
    velocity: Num,
) -> tuple[float, trajectory_pb2.SE3Trajectory]:
    """
    Creates an arm trajectory from a series of poses and the velocity that the poses should be followed at.

    Returns:
        full_time: the time (in seconds) that the returned trajectory command is expected to take
        trajectory: the trajectory made from the given poses
    """
    prev_pose = None
    last_t = 0
    points = []
    full_time = 0

    # Create a trajectory from the points
    for pose in poses:
        if prev_pose is None:
            time_in_sec = 0
            seconds = int(0)
            nanos = int(0)
        else:
            # Compute the distance from the current hand position to the new hand position
            dist = math.sqrt(
                  ((prev_pose.x - pose.x) ** 2)
                + ((prev_pose.y - pose.y) ** 2)
                + ((prev_pose.z - pose.z) ** 2)
            )
            time_in_sec = dist / velocity
            full_time += time_in_sec
            seconds = int(time_in_sec + last_t)
            nanos = int((time_in_sec + last_t - seconds) * 1e9)

        position = geometry_pb2.Vec3(x=pose.x, y=pose.y, z=pose.z)
        rotation = geometry_pb2.Quaternion(
            w=pose.rot.w, x=pose.rot.x, y=pose.rot.y, z=pose.rot.z
        )
        this_se3_pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)

        points.append(
            trajectory_pb2.SE3TrajectoryPoint(
                pose=this_se3_pose,
                time_since_reference=duration_pb2.Duration(
                    seconds=seconds, nanos=nanos
                ),
            )
        )

        prev_pose = pose
        last_t = time_in_sec + last_t

    hand_trajectory = trajectory_pb2.SE3Trajectory(points=points)

    return (full_time, hand_trajectory)

