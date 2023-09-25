//
// Created by milinda on 7/25/17.
/**
*@author Milinda Fernando
*School of Computing, University of Utah
*@brief Header file for the GR simulation.
*/
//

#ifndef SFCSORTBENCH_GR_H
#define SFCSORTBENCH_GR_H

#include <iostream>
#include <vector>

#include "dendroProfileParams.h"

#define Rx (em3::EM3_COMPD_MAX[0]-em3::EM3_COMPD_MIN[0])
#define Ry (em3::EM3_COMPD_MAX[1]-em3::EM3_COMPD_MIN[1])
#define Rz (em3::EM3_COMPD_MAX[2]-em3::EM3_COMPD_MIN[2])

#define RgX (em3::EM3_OCTREE_MAX[0]-em3::EM3_OCTREE_MIN[0])
#define RgY (em3::EM3_OCTREE_MAX[1]-em3::EM3_OCTREE_MIN[1])
#define RgZ (em3::EM3_OCTREE_MAX[2]-em3::EM3_OCTREE_MIN[2])

#define GRIDX_TO_X(xg) (((Rx/RgX)*(xg-em3::EM3_OCTREE_MIN[0]))+em3::EM3_COMPD_MIN[0])
#define GRIDY_TO_Y(yg) (((Ry/RgY)*(yg-em3::EM3_OCTREE_MIN[1]))+em3::EM3_COMPD_MIN[1])
#define GRIDZ_TO_Z(zg) (((Rz/RgZ)*(zg-em3::EM3_OCTREE_MIN[2]))+em3::EM3_COMPD_MIN[2])

#define X_TO_GRIDX(xc) (((RgX/Rx)*(xc-em3::EM3_COMPD_MIN[0]))+em3::EM3_OCTREE_MIN[0])
#define Y_TO_GRIDY(yc) (((RgY/Ry)*(yc-em3::EM3_COMPD_MIN[1]))+em3::EM3_OCTREE_MIN[1])
#define Z_TO_GRIDZ(zc) (((RgZ/Rz)*(zc-em3::EM3_COMPD_MIN[2]))+em3::EM3_OCTREE_MIN[2])

#endif //SFCSORTBENCH_GR_H
