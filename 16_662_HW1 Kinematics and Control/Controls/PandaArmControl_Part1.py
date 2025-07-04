import mujoco as mj
from mujoco import viewer
import numpy as np
import math
import quaternion


# Set the XML filepath
xml_filepath = "../franka_emika_panda/panda_nohand_torque_fixed_board.xml"

################################# Control Callback Definitions #############################

# Control callback for gravity compensation
def gravity_comp(model, data):
    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints

    data.ctrl[:7] = data.qfrc_bias[:7]

# Force control callback
def force_control(model, data):  # TODO:
    # Implement a force control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Get the Jacobian for the desired location on the robot (The end-effector)

    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables
    jp = np.zeros((3,model.nv))
    jr = np.zeros((3,model.nv))

    mj.mj_jac(model, data, jp, jr, body.xpos,body.id)

    J = np.vstack((jp, jr))

    # Specify the desired force in global coordinates
    force_des = np.array([15, 0, 0, 0, 0, 0])

    # Compute the required control input using desied force values
    torque_des= J.T @ force_des


    # Set the control inputs
    data.ctrl[:7] = torque_des +data.qfrc_bias[:7] 

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Force readings updated here
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]

# Control callback for an impedance controller
def impedance_control(model, data):  # TODO:

    # Implement an impedance control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.
    # tune these
    kp = 10
    kd = 5

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired position
    pos_des = np.array([15, 0, 0])/kp + body.xpos

    # Set the desired velocities
    pos_error = pos_des - body.xpos


    # Set the desired orientation (Use numpy quaternion manipulation functions)

    # Get the current orientation

    # Get orientation error

    # Get the position error

    # Get the Jacobian at the desired location on the robot

    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables

    jp = np.zeros((3,model.nv))
    jr = np.zeros((3,model.nv))


    mj.mj_jac(model, data, jp, jr, body.xpos,body.id)

    J = np.vstack((jp, jr))

    # Compute the impedance control input torques
    torques = J.T @ (kp* (np.concatenate((pos_error, np.array([0,0,0])))) +kd*(-body.cvel))


    # Set the control inputs
    data.ctrl[:7] = torques+ data.qfrc_bias[:7]


    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Update force sensor readings
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]


def position_control(model, data):
    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    desired_joint_positions = np.array(
         [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])

    # Desired gain on position error (K_p)
    Kp = 1000

    # Desired gain on velocity error (K_d)
    Kd = 1000

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp * \
        (desired_joint_positions-data.qpos[:7]) + Kd * \
        (np.array([0, 0, 0, 0, 0, 0, 0])-data.qvel[:7])


####################################### MAIN #####################################
if __name__ == "__main__":
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF

    #mj.set_mjcb_control(gravity_comp)  # TODO:

    #mj.set_mjcb_control(force_control)
    #mj.set_mjcb_control(impedance_control)
    #mj.set_mjcb_control(position_control)

    ################################# Swap Callback Above This Line #################################

    # Initialize variables to store force and time data points
    force_sensor_max_time = 10
    force = np.zeros(int(force_sensor_max_time/model.opt.timestep))
    time = np.linspace(0, force_sensor_max_time, int(
        force_sensor_max_time/model.opt.timestep))

    # Launch the simulate viewer
    viewer.launch(model, data)

    # Save recorded force and time points as a csv file
    # force = np.reshape(force, (5000, 1))
    # time = np.reshape(time, (5000, 1))
    # plot = np.concatenate((time, force), axis=1)
    # np.savetxt('impedance_vs_time_31.csv', plot, delimiter=',')
