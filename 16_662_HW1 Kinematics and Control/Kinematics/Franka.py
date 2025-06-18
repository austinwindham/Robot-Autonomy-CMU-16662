import numpy as np
import RobotUtil as rt
import math

class FrankArm:
    def __init__(self):
        # Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
        self.Rdesc = [
            [0, 0, 0, 0., 0, 0.333],  # From robot base to joint1
            [-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
            [np.pi/2, 0, 0, 0.0825, 0, 0],
            [-np.pi/2, 0, 0, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107]  # From joint5 to end-effector center
        ]

        # Define the axis of rotation for each joint
        self.axis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]

        # Set base coordinate frame as identity - NOTE: don't change
        self.Tbase = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        # Initialize matrices - NOTE: don't change this part
        self.Tlink = []  # Transforms for each link (const)
        self.Tjoint = []  # Transforms for each joint (init eye)
        self.Tcurr = []  # Coordinate frame of current (init eye)
        
        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(
                self.Rdesc[i][0:3], self.Rdesc[i][3:6]))
            self.Tcurr.append([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.], [0, 0, 0, 1]])
            self.Tjoint.append([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]])

        self.Tlinkzero = rt.rpyxyz2H(self.Rdesc[0][0:3], self.Rdesc[0][3:6])

        self.Tlink[0] = np.matmul(self.Tbase, self.Tlink[0])

        # initialize Jacobian matrix
        self.J = np.zeros((6, 7))

        self.q = [0., 0., 0., 0., 0., 0., 0.]
        self.ForwardKin([0., 0., 0., 0., 0., 0., 0.])

    def ForwardKin(self, ang):
        '''
        inputs: joint angles
        outputs: joint transforms for each joint, Jacobian matrix
        '''

        self.q[0:-1] = ang

        # Compute current joint and end effector coordinate frames (self.Tjoint). Remember that not all joints rotate about the z axis!
        # follow slides

        for i in range(len(self.Rdesc)):

            if i == 0:
                self.Tcurr[i] = self.Tlink[i] @ [[math.cos(self.q[i]), -math.sin(self.q[i]), 0,0],
                                                 [math.sin(self.q[i]), math.cos(self.q[i]), 0,0],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]]

            else:
                self.Tcurr[i] = self.Tcurr[i-1] @ self.Tlink[i] @ [[math.cos(self.q[i]), -math.sin(self.q[i]), 0,0],
                                                                   [math.sin(self.q[i]), math.cos(self.q[i]), 0,0],
                                                                   [0, 0, 1, 0],
                                                                   [0, 0, 0, 1]]

        for i in range(len(self.Rdesc)-1):

            cross_product_J =np.cross(self.Tcurr[i][:3,2], self.Tcurr[7][:3,3] -self.Tcurr[i][:3,3])                                                   
            self.J[:, i] = np.concatenate((cross_product_J, self.Tcurr[i][:3,2]))



        return self.Tcurr, self.J

    def IterInvKin(self, ang, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        inputs: starting joint angles (ang), target end effector pose (TGoal)

        outputs: computed joint angles to achieve desired end effector pose, 
        Error in your IK solution compared to the desired target
        '''
        # C and W from paper
        W = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]]
        
        C = [[1000000.0, 0.0, 0.0,0.0, 0.0, 0.0],
             [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0],
             [0.0,0.0, 0.0, 1000.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 1000.0, 0.0],
             [0.0, 0.0,0.0, 0.0, 0.0, 1000.0]]

        Err = 0
        q = ang

        ang_step = 0.1 
        norm_step = 0.01

        while True:
            
            # get current
            self.ForwardKin(q)
            r_curr =self.Tcurr[-1][:3,:3]

            
            rot_err = TGoal[:3,:3] @ r_curr.T
            rot_err_axis, rot_ang = rt.R2axisang(rot_err)

            if rot_ang >ang_step:
                rot_ang = ang_step

            if rot_ang < -ang_step:
                rot_ang = -ang_step

            rot_err_axis = np.array(rot_err_axis)
            r_err = rot_ang*rot_err_axis

            # get translation error
            t_err = TGoal[:3,3]-self.Tcurr[-1][:3,3]

            if np.linalg.norm(t_err) >norm_step:
                t_err = t_err*(norm_step/np.linalg.norm(t_err))

            
            Err = np.concatenate((t_err, r_err), axis = 0)
            
            # pseudo inverse steps
            J_ps_inv = np.linalg.inv(W) @ self.J.T @np.linalg.inv(self.J @np.linalg.inv(W) @self.J.T +np.linalg.inv(C))
            qt1 = q+ J_ps_inv@Err
            q = qt1

            if np.linalg.norm(Err[0:3])< x_eps and np.linalg.norm(Err[3:6])<r_eps:
                break

        return self.q[0:-1], Err
