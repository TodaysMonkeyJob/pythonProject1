"""
Gas sensor drivers for the Raspberry Pi (A7 / TOD-48): automatic detection on I2C / serial, readings in each sensor's
native unit. See ``docs/sensors.md`` and ``drivers/gas/discovery.py``.
"""
from drivers.gas.electrochemical import DFRobotGravity, DFRobotGravityUART
from drivers.gas.graphene_fet import GrapheneFET
from drivers.gas.index_only import ENS160
from drivers.gas.mos_analog import MQAds1115
from drivers.gas.particle import PMS5003, SDS011, SPS30

DRIVERS = {d.name: d for d in (GrapheneFET, DFRobotGravity, DFRobotGravityUART, MQAds1115, PMS5003, SDS011, SPS30,
                               ENS160)}
