Guide to write an asset
-----------------------

To write an asset that will be compatible with the simulator, it needs to be given a parent node, a name and to inherit the class Asset and execute the Asset constructor in the new asset constructor:
```python
class NewAsset(Asset):
    def __init__(self, node, name):
        Asset.__init__(self, node, name)
```

## Asset variables

The Asset class has 5 variables:

* self.parent_node: The node that includes the asset. All variables from the parent node such as the environment or the microgrid can therefore be accessed by the asset through this variable.
* self.name: The name of the asset. Used to identify the asset. Used for example to plot the asset.
* self.power_limit_high: Maximum production power of asset. Updated at every simulation step. Equals 0 if the asset can not produce power and positive when the asset can produce power. Should not be negative.
* self.power_limit_low: Minimum production power of asset. Updated at every simulation step. Equals 0 if the asset can not receive power and negative when the asset can receive power. Should not be positive.
* self.power_output: Power production of asset for a simulation step. Updated at every simulation step. Negative when drawing power from other assets and positive when providing power to other assets.

## Asset functions

The Asset class has 2 functions that define how assets work:

* set_power_limits(self, time_step, environment): This function updates at each simulation step the values of self.power_limit_high and self.power_limit_low based on the time length of the simulation step (time_step), the environment and any other variables internal to the implemented asset. This step is done for all assets after the environment is updated and before any actions are executed by the energy management system. Updating the limits for each asset gives the information to the energy management system on what actions are possible.
* power_consequences(self, time_step, reward): This function applies all the consequences to the asset for functionning at a defined power (self.power_output) for the time length of the simulation step (time_step). This also includes changing the reward given as a function parameter. This step is done for all assets after every actions are executed by the energy management system and before the environment is updated for the next simulation step.

## Summary and remarks

To summarize: 
* set_power_limits(...) updates self.power_limit_low and self.power_limit_high. 
* Based on this update, the energy management system defines self.power_output for each asset.
* power_consequences(...) defines the consequences applied to the asset and updates the reward based on self.power_output.

In your new asset class, you can write extra functions (static or not), instance variables, ... Just make sure to follow the coding guidelines from [the repository README](../../README.md) and try to make it as consistent with the other asset classes as possible.
