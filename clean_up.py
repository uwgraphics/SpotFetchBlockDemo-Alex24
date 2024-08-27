import asyncio
import time
from typing import Any
from PIL.Image import Image
from PIL.ImageDraw import ImageDraw
import numpy as np
import torch

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import geometry_pb2
from bosdyn.api import manipulation_api_pb2
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers
from bosdyn.client.math_helpers import SE3Pose
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    block_for_trajectory_cmd,
    block_until_arm_arrives
)

from spot_api import Spot


async def main():
    """The main method of the demo."""

    async def find_block(spot: Spot, cam: str | None = None) -> None | tuple[str, float, tuple[int, int, int, int], Image, Any]:
        """
        Attempts to find any blocks with the given camera and returns None if none are found.
        """

        # capture an image
        (rgb, _greyscale, _depth) = await spot.get_images(
            frontleft=True, #frontright=True, #left=True, right=True, back=True,
            rgb=True, #depth=True,
            rotate=True
        )

        best_block_cam_name   = None
        best_block_conf       = -np.inf
        best_block_box        = None
        best_block_reponse    = None
        best_block_rgb_img = None

        # check to see whether an image contains the block
        for (cam_name, rgb_image, response) in zip([c for c in ["frontleft"] if cam is None or c == cam], rgb.image_list(), rgb.response_list()):
            result = yolo_model(rgb_image)

            #print(response.shot.transforms_snapshot)

            # for every found block, check to make sure that we are sure enough that it is a block and that it is the most sure block (so far)
            for block_info in result.xyxy[0].cpu():
                block_box = block_info[0:4] # the [left, top, right, bottom] axis-aligned bounding box of the block
                block_confidence = block_info[4] # how sure we are that this is the box

                if block_confidence <= MIN_SURE or block_confidence <= best_block_conf:
                    # not sure enough
                    continue

                best_block_cam_name   = cam_name
                best_block_conf       = block_confidence
                best_block_box        = block_box
                best_block_reponse    = response
                best_block_rgb_img    = rgb_image

        if best_block_cam_name is None:
            return None

        return (best_block_cam_name, best_block_conf, best_block_box, best_block_rgb_img, best_block_reponse)

    def compute_stand_location_and_yaw(vision_tform_target: SE3Pose | geometry_pb2.SE3Pose, robot_state_client: RobotStateClient, distance_margin: float):

        if isinstance(vision_tform_target, geometry_pb2.SE3Pose):
            vision_tform_target = SE3Pose.from_proto(vision_tform_target)

        # Compute drop-off location:
        #   Draw a line from Spot to the person
        #   Back up 2.0 meters on that line

        vision_tform_robot = frame_helpers.get_a_tform_b(
            robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
            frame_helpers.VISION_FRAME_NAME, frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)

        # Compute vector between robot and person
        robot_rt_person_ewrt_vision = [
            vision_tform_robot.x - vision_tform_target.x, vision_tform_robot.y - vision_tform_target.y,
            vision_tform_robot.z - vision_tform_target.z
        ]

        # Compute the unit vector.
        if np.linalg.norm(robot_rt_person_ewrt_vision) < 0.01:
            robot_rt_person_ewrt_vision_hat = vision_tform_robot.transform_point(1, 0, 0)
        else:
            robot_rt_person_ewrt_vision_hat = robot_rt_person_ewrt_vision / np.linalg.norm(
                robot_rt_person_ewrt_vision)

        # Starting at the person, back up meters along the unit vector.
        drop_position_rt_vision = [
            vision_tform_target.x + robot_rt_person_ewrt_vision_hat[0] * distance_margin,
            vision_tform_target.y + robot_rt_person_ewrt_vision_hat[1] * distance_margin,
            vision_tform_target.z + robot_rt_person_ewrt_vision_hat[2] * distance_margin
        ]

        # We also want to compute a rotation (yaw) so that we will face the person when dropping.
        # We'll do this by computing a rotation matrix with X along
        #   -robot_rt_person_ewrt_vision_hat (pointing from the robot to the person) and Z straight up:
        xhat = -robot_rt_person_ewrt_vision_hat
        zhat = [0.0, 0.0, 1.0]
        yhat = np.cross(zhat, xhat)
        mat = np.matrix([xhat, yhat, zhat]).transpose()
        heading_rt_vision = math_helpers.Quat.from_matrix(mat).to_yaw()

        return drop_position_rt_vision, heading_rt_vision

    def get_walking_params(max_linear_vel, max_rotation_vel):
        max_vel_linear = geometry_pb2.Vec2(x=max_linear_vel, y=max_linear_vel)
        max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear, angular=max_rotation_vel)
        vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
        params = RobotCommandBuilder.mobility_params()
        params.vel_limit.CopyFrom(vel_limit)
        return params

    MIN_SURE = 0.7

    # create the Spot robot
    spot = Spot()

    # load model used to identify block to clean up
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./models/yolov5_graphics_block_detector.pt')

    idle_rotate_right = True # if False, then Spot rotates to the left when idle instead

    holding_block = False

    CAN_MOVE      = True
    TURN_VELOCITY = 0.2 # radians/s
    TURN_TIME     = 0.5 # seconds
    #MOVE_VELOCITY = 0.2 # m/s
    #MOVE_TIME     = 0.5 # seconds
    SLEEP_TIME    = 0.1 # seconds

    # get a lease on Spot so that we can move Spot around
    with spot.leased(take=True) as spot:

        # power spot on if not powered on already
        if not await spot.are_motors_powered_on():
            await spot.power_motors_on()

        #await spot.stow_arm()

        # Spot should stand up if it is not already standing
        await spot.stand()

        # stow the arm if it is not already stowed
        await spot.stow_arm()

        # main loop
        while True:

            if holding_block:
                # since we are holding a block, take it to the nearest fiducial

                # find fiducial
                while True:
                    fiducials = await spot.fiducials_in_view()

                    if len(fiducials) == 0:
                        # no fiducial so idly turn until we find one
                        if CAN_MOVE:
                            if idle_rotate_right:
                                print("Idle rotate right")
                                await spot.rotate_right(TURN_VELOCITY, TURN_TIME)
                            else:
                                print("Idle rotate left")
                                await spot.rotate_left(TURN_VELOCITY, TURN_TIME)
                        else:
                            print("Idle Sleeping")
                            time.sleep(SLEEP_TIME)
                        continue

                    break

                #state = spot.state_client().get_robot_state().kinematic_state.transforms_snapshot
                #body_t_hand       = SE3Pose.from_proto(state.child_to_parent_edge_map["hand"].parent_tform_child)
                #body_t_vision     = SE3Pose.from_proto(state.child_to_parent_edge_map["vision"].parent_tform_child)
                vision_t_ffiducial = SE3Pose.from_proto(fiducials[0].transforms_snapshot.child_to_parent_edge_map["filtered_fiducial_1"].parent_tform_child)

                #body_t_ffiducial = body_t_vision * vision_t_ffiducial
                #hand_t_ffiducial = body_t_hand.inverse() * body_t_ffiducial

                drop_position_t_vision, heading_t_vision = compute_stand_location_and_yaw(
                    vision_t_ffiducial,
                    spot.state_client(),
                    distance_margin=1.0
                )

                wait_t_vision, wait_heading_t_vision = compute_stand_location_and_yaw(
                    vision_t_ffiducial,
                    spot.state_client(),
                    distance_margin=2.0
                )

                # Limit the speed so we don't charge at the person.
                se2_pose = geometry_pb2.SE2Pose(
                    position=geometry_pb2.Vec2(x=drop_position_t_vision[0],
                                            y=drop_position_t_vision[1]), angle=heading_t_vision)
                move_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                    se2_pose, frame_name=frame_helpers.VISION_FRAME_NAME,
                    params=get_walking_params(0.5, 0.5))
                end_time = 5.0
                cmd_id = spot.command_client().robot_command(command=move_cmd,
                                                    end_time_secs=time.time() + end_time)

                # Wait until the robot repots that it is at the goal.
                block_for_trajectory_cmd(spot.command_client(), cmd_id, timeout_sec=10)

                print('Arrived at goal, dropping object...')

                # Do an arm-move to gently put the object down.
                # Build a position to move the arm to (in meters, relative to
                # and expressed in the gravity aligned body frame).
                x = 0.75
                y = 0
                z = -0.25
                hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

                # Point the hand straight down with a quaternion.
                qw = 0.707
                qx = 0
                qy = 0.707
                qz = 0
                flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

                flat_body_tform_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                            rotation=flat_body_Q_hand)

                robot_state = spot.state_client().get_robot_state()
                vision_tform_flat_body = frame_helpers.get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)

                vision_tform_hand_at_drop = vision_tform_flat_body * math_helpers.SE3Pose.from_proto(flat_body_tform_hand)

                # duration in seconds
                seconds = 1

                arm_command = RobotCommandBuilder.arm_pose_command(
                    vision_tform_hand_at_drop.x, vision_tform_hand_at_drop.y,
                    vision_tform_hand_at_drop.z, vision_tform_hand_at_drop.rot.w,
                    vision_tform_hand_at_drop.rot.x, vision_tform_hand_at_drop.rot.y,
                    vision_tform_hand_at_drop.rot.z, frame_helpers.VISION_FRAME_NAME, seconds)

                # Keep the gripper closed.
                gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)

                # Combine the arm and gripper commands into one RobotCommand
                command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

                # Send the request
                cmd_id = spot.command_client().robot_command(command)

                # Wait until the arm arrives at the goal.
                block_until_arm_arrives(spot.command_client(), cmd_id)

                # Open the gripper
                gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
                command = RobotCommandBuilder.build_synchro_command(gripper_command)
                cmd_id = spot.command_client().robot_command(command)

                # Wait for the block to fall out
                time.sleep(1.5)

                # Stow the arm.
                stow_cmd = RobotCommandBuilder.arm_stow_command()
                spot.command_client().robot_command(stow_cmd)

                time.sleep(1)

                print('Backing up and waiting...')

                # Back up one meter
                se2_pose = geometry_pb2.SE2Pose(
                    position=geometry_pb2.Vec2(x=wait_t_vision[0],
                                            y=wait_t_vision[1]),
                    angle=wait_heading_t_vision)
                move_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                    se2_pose, frame_name=frame_helpers.VISION_FRAME_NAME,
                    params=get_walking_params(0.5, 0.5))
                end_time = 5.0
                cmd_id = spot.command_client().robot_command(command=move_cmd,
                                                    end_time_secs=time.time() + end_time)

                # Wait until the robot reports that it is at the goal.
                block_for_trajectory_cmd(spot.command_client(), cmd_id, timeout_sec=5)

                #sec = 10
                #print(f"Sleeping {sec}...")
                #time.sleep(sec)
                #print(f"...Done Sleeping {sec}")
                break

            # --- Look For The Block ---

            # Find any blocks

            block = await find_block(spot)

            if block is None:
                if CAN_MOVE:
                    if idle_rotate_right:
                        print("Idle rotate right")
                        await spot.rotate_right(TURN_VELOCITY, TURN_TIME)
                    else:
                        print("Idle rotate left")
                        await spot.rotate_left(TURN_VELOCITY, TURN_TIME)
                else:
                    print("Idle Sleeping")
                    time.sleep(SLEEP_TIME)
                continue

            # found block

            (cam_name, conf, (left, top, right, bottom), img, response) = block
            image_width = img.width
            image_height = img.height

            #print("Snapshot:")
            #print(response.shot.transforms_snapshot)

            # --- Move Towards Block ---

            # get center pixel of block in image
            cx = int(np.round(left + np.abs(right  - left)/2))
            cy = int(np.round(top  + np.abs(bottom - top )/2))

            # --- Pick Up The Block ---

            # show where block is in image
            draw = ImageDraw(img)
            draw.line((cx, cy, cx, cy), fill="red", width=10)
            draw.rectangle((left, top, right, bottom), outline="green", width=2)
            img.show()

            # Our machine learning model assumes that the image has been rotated 
            # but the Spot API wants a pixel from the unrotated image. As such, we
            # must unrotate the pixel so that it is as Spot expects it

            # Spot expects (0, 0) in bottom-right of image but it is currently bottom left, so invert the x axis
            cx = image_width - cx
            # unrotate the pixels
            match np.round(spot.BODYCAM_ROTATION[cam_name] % 360):
                case 0:
                    rimg_width, rimg_height = image_width, image_height
                    rcx, rcy = cx, cy
                case 180 | -180:
                    rimg_width, rimg_height = image_width, image_height
                    rcx, rcy = cx, cy
                case 90 | -270:
                    rimg_width, rimg_height = image_height, image_width
                    rcx, rcy = cy, cx
                case 270 | -90:
                    rimg_width, rimg_height = image_height, image_width
                    rcx, rcy = cy, cx
                case rot:
                    raise ValueError(f"Camera had unexpected rotation {rot}")
            cx = image_width - cx # put x back where it was before the rotation code
                

#                # images rotated by 0, 90, 180, or 270 degrees so we only have to switch the axis if they are rotated
#                if np.round(np.abs(spot.BODYCAM_ROTATION[cam_name]) % 360) in [90, 270]:
#                    #rimg_width, rimg_height = image_height, image_width
#                    rcx, rcy = cy, cx
#                else:
#                    #rimg_width, rimg_height = image_width, image_height
#                    rcx, rcy = cx, cy

            #print(rcx, rcy, rimg_width, rimg_height)

            # create the command

            # for command, y is left and x is down so 0,0 in image is top-right
            # <---yx
            #      |
            #     \|/
            #      '
            grasp = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=geometry_pb2.Vec2(
                    x=rcx,
                    y=rcy,
                ),
                transforms_snapshot_for_camera=response.shot.transforms_snapshot,
                frame_name_image_sensor=response.shot.frame_name_image_sensor,
                camera_model=response.source.pinhole,
                walk_gaze_mode=manipulation_api_pb2.PICK_AUTO_WALK_AND_GAZE,
                #walk_gaze_mode=manipulation_api_pb2.PICK_AUTO_GAZE,
            )

            # We can specify where in the gripper we want to grasp. About halfway is generally good for
            # small objects like this. For a bigger object like a shoe, 0 is better (use the entire
            # gripper)
            grasp.grasp_params.grasp_palm_to_fingertip = 0.5

            if (constraint_grip := False):
                # Tell the grasping system that we want a top-down grasp.

                # Add a constraint that requests that the x-axis of the gripper is pointing in the
                # negative-z direction in the vision frame.

                # The axis on the gripper is the x-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

                # The axis in the vision frame is the negative z-axis
                axis_to_align_with_ewrt_vision = geometry_pb2.Vec3(x=0, y=0, z=-1)

                # Add the vector constraint to our proto.
                constraint = grasp.grasp_params.allowable_orientation.add()
                constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
                    axis_on_gripper_ewrt_gripper
                )
                constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
                    axis_to_align_with_ewrt_vision
                )

                # We'll take anything within about 15 degrees for top-down or horizontal grasps.
                constraint.vector_alignment_with_tolerance.threshold_radians = 0.25

                # Specify the frame we're using.
                grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME
                grasp.grasp_params.grasp_params_frame_name = frame_helpers.BODY_FRAME_NAME

            # Build the proto
            grasp_request = manipulation_api_pb2.ManipulationApiRequest(
                pick_object_in_image=grasp
            )

            stow_cmd = RobotCommandBuilder.arm_stow_command()
            spot.command_client().robot_command(stow_cmd)

            # Send the request (will walk to the object to pick it up)
            print('Grasping block')
            cmd_response = spot.manipulation_api_client().manipulation_api_command(
                manipulation_api_request=grasp_request,
            )

            # -- Wait for Grasp to Finish --

            grasp_done = False
            failed = False
            time_start = time.time()
            while not grasp_done:

                feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=cmd_response.manipulation_cmd_id
                )

                # Send a request for feedback
                response = spot.manipulation_api_client().manipulation_api_feedback_command(
                    manipulation_api_feedback_request=feedback_request
                )

                current_state = response.current_state
                current_time = time.time() - time_start
                print(
                    'Current state ({time:.1f} sec): {state}'.format(
                        time=current_time,
                        state=manipulation_api_pb2.ManipulationFeedbackState.Name(
                            current_state
                        )
                    )
                )

                failed_states = [
                    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                    manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
                    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
                    manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE
                ]

                failed = current_state in failed_states
                grasp_done = current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or failed

                time.sleep(SLEEP_TIME)

            holding_block = not failed

            # carry object
            #await spot.carry()

            carry_cmd = RobotCommandBuilder.arm_carry_command()
            carry_cmd = RobotCommandBuilder.build_synchro_command(carry_cmd)
            spot.command_client().robot_command(carry_cmd)
            time.sleep(1)

        # spot can sit back down now
        await spot.sit()



if __name__ == "__main__":
    asyncio.run(main())
