# Moves deer when vehicle is within distance sensor
from controller import Supervisor

timestep = 32

robot = Supervisor()

distSensor = robot.getDevice('distance sensor')
distSensor.enable(timestep)
deer = robot.getFromDef('DEER')
translation_field = deer.getField('translation')

move = False

x = translation_field.getSFVec3f()[0]
while robot.step(timestep) != -1:
    if distSensor.getValue() < 10:
        move = True
    if move:
        translation_field.setSFVec3f([x, translation_field.getSFVec3f()[1], translation_field.getSFVec3f()[2]])
        x += 0.2 * (timestep / 1000.0)
    if x > 10:
        move = False
