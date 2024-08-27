
# Fetch Block Demo

This is a small demo that makes a Boston Dynamics Spot robot spin slowly until it sees a blue block in the world, pick up the blue block it sees, and then bring the blue block to a nearby fiducial. The demo uses AI to recognize the block in the environment and built-in fiducial tracking to find and bring the block to the fiducial.

## Quick Start

In order to use this repository, you will first need to clone (via `git clone https://github.com/boston-dynamics/spot-sdk.git` or a similar command) the Spot SDK from boston dynamics and put it at the top level of this repo.

The repo will then look something like this:
```text
./spot-sdk/** # The downloaded Spot SDK
./spot_api.py # This repo's code layer over the SDK (to make the SDK easier to use)
./clean_up.py # The spot block demo.
... # The rest of the directories and files
```

Pip install all of the other dependencies found in the "Dependencies" section of this README.

Connect to Spot's wifi.

Make sure that Spot can power up its motors i.e. that the hardware button on Spot is not pressed to prevent Spot from powering up.

Pick up the touchpad provided with Spot and use it to drive Spot to a suitable location for the demo. A blue block (refer to `./block_images/block_images_color` to figure out what such a block should look like) and a fiducial should be around Spot when the demo is started. If Spot does not immediately see a blue block then it will rotate in place until it sees one (this means that you *can* put the block down after the demo has started but be careful as the robot will immediately go for it as soon as it sees it on the ground).

WARNING: Stay at least 6 feet away from spot, the blue block, and the fiducial at all times during the demo. Failing to do so may result in serious injury as **Spot's collision detection is turned off at certain points during the demo**.

IMPORTANT: Spot's tablet functions as it's estop so be prepared to use the tablet to cut Spot's power if it is about to do something dangerous such as run into a person or wall. You can alternatively use the tablet to hijack (take back control of) the robot which will both stop the demo and allow you to manually pilot the robot back to a more suitable location. It is recommended to hijack the robot whenever possible as hijacking does not cut off Spot's power and therefore does not make Spot violently fall to the ground as cutting off its power does.

At this point, running `python spot_sweep_touch.py` (or running any of the other demos in a similar manner) should prompt you to input your username and password for Spot. After inputting your username and password, the demo should run and make Spot do as described in the `Demos` section of this README.

## Dependencies

 - The Boston Dynamics Spot SDK
 - numpy
 - pytorch
 - PIL

## Demos
 - `./clean_up.py` A small demo that picks up a block on the ground and puts it in another location designated by a fiducial.

## Other Files
 - `./spot_api.py` The Spot SDK provides many methods to control Spot, but in a way that is often confusing and untyped. As such, this file consolidates much of the SDK's functionality into a higher-level API that is easier to use and understand.


