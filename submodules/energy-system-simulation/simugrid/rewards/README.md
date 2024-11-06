Guide to write a reward calculator
-----------------------

## Reward constructor

To write an new reward calculator, first create a new python file in this folder then you will need to define a name (in the example below `NewReward`) and write the constructor as follows:
```python
from simugrid.rewards.reward import Reward

class NewReward(Reward):
    def __init__(self, list_KPI):
        Reward.__init__(self, list_KPI)
```
`list_KPI` is list of names that defines which reward will be evaluated. For example if list_KPI=\["OPEX", "CAPEX"\] for a simulation, then the reward calculatore will only evaluate the CAPEX and OPEX.

## Reward variables

The Reward class has 2 variables:

* `self.reward_information`: List containing all the reward information from assets for the current timestep
* `self.KPIs`: Name and value of all the KPIs to evaluate

## KPI calculation

To  calculate the new KPI values, the function `calculate_kpi` needs to be implemented. In this function you can decide to either select the default function to calculate the wanted KPI or choose to calculate the KPI the way you want for a specific asset based on its RewardInformation object. This function is called once at each time step.

You can see an example of a custom Reward object in [this file](example.py).

## Reward Information object

Object storing the information needed to calculate the different KPIs at each timestep. It contains an asset object, the power produced by the asset and the environment of the asset.
