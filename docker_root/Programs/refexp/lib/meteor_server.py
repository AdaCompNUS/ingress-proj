import numpy as np
import rospy

import ingress_msgs.srv

from meteor_score import Meteor

class MeteorServer(object):

    def m_score(self, req):
        score = self.meteor.score(req.ref, req.tar)
        return ingress_msgs.srv.MeteorScoreResponse(score)

    def __init__(self):
        rospy.init_node('meteor_score_server')
        self.meteor = Meteor()
        s = rospy.Service('meteor_score', ingress_msgs.srv.MeteorScore, self.m_score)

        print "Meteor server ready!"
        rospy.spin()


if __name__ == "__main__":
    ms = MeteorServer()

