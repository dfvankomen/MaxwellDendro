/**
 * @file em3Ctx.cpp
 * @author Milinda Fernando (milinda@cs.utah.edu)
 * @brief EM3 Ctx class.
 * @version 0.1
 * @date 2020-07-18
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "em3Ctx.h"
namespace em3 {
EM3Ctx::EM3Ctx(ot::Mesh* pMesh) : Ctx() {
    m_uiMesh = pMesh;

    m_var[VL::CPU_EV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, EM3_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_OUT].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_NUM_VARS, true);

    m_var[VL::CPU_CV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, EM3_CONSTRAINT_NUM_VARS,
                                    true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_CONSTRAINT_NUM_VARS, true);

    m_var[VL::CPU_PV].create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, EM3_NUM_VARS, true);

    m_uiTinfo._m_uiStep = 0;
    m_uiTinfo._m_uiT = 0;
    m_uiTinfo._m_uiTb = EM3_RK45_TIME_BEGIN;
    m_uiTinfo._m_uiTe = EM3_RK45_TIME_END;
    m_uiTinfo._m_uiTh = EM3_RK45_TIME_STEP_SIZE;

    m_uiElementOrder = EM3_ELE_ORDER;

    m_uiMinPt = Point(EM3_GRID_MIN_X, EM3_GRID_MIN_Y, EM3_GRID_MIN_Z);
    m_uiMaxPt = Point(EM3_GRID_MAX_X, EM3_GRID_MAX_Y, EM3_GRID_MAX_Z);

    deallocate_em3_deriv_workspace();
    allocate_em3_deriv_workspace(m_uiMesh, 1);

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, EM3_NUM_VARS,
                                      EM3_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, EM3_NUM_VARS,
                                    EM3_ASYNC_COMM_K);

    return;
}

EM3Ctx::~EM3Ctx() {
    for (unsigned int i = 0; i < VL::END; i++) m_var[i].destroy_vector();

    deallocate_em3_deriv_workspace();
    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, EM3_NUM_VARS,
                                      EM3_ASYNC_COMM_K);
}

int EM3Ctx::rhs(DVec* in, DVec* out, unsigned int sz, DendroScalar time) {
    // all the variables should be packed together.
    // assert(sz == 1);
    // DendroScalar** sVar;
    // in[0].Get2DArray(sVar, false);

    this->unzip(*in, m_var[VL::CPU_EV_UZ_IN], EM3_ASYNC_COMM_K);

#ifdef __PROFILE_CTX__
    this->m_uiCtxpt[ts::CTXPROFILE::RHS].start();
#endif

    DendroScalar* unzipIn[EM3_NUM_VARS];
    DendroScalar* unzipOut[EM3_NUM_VARS];

    m_var[CPU_EV_UZ_IN].to_2d(unzipIn);
    m_var[CPU_EV_UZ_OUT].to_2d(unzipOut);

    const ot::Block* blkList = m_uiMesh->getLocalBlockList().data();
    const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();

    em3DoRHS(unzipOut, (const DendroScalar**)unzipIn, blkList, numBlocks);

#ifdef __PROFILE_CTX__
    this->m_uiCtxpt[ts::CTXPROFILE::RHS].stop();
#endif

    this->zip(m_var[CPU_EV_UZ_OUT], *out);

    return 0;
}

int EM3Ctx::initialize() {
    if (EM3_RESTORE_SOLVER) {
        this->restore_checkpt();
        return 0;
    }

    this->init_grid();

    bool isRefine = false;
    DendroIntL oldElements, oldElements_g;
    DendroIntL newElements, newElements_g;

    DendroIntL oldGridPoints, oldGridPoints_g;
    DendroIntL newGridPoints, newGridPoints_g;

    unsigned int iterCount = 1;
    const unsigned int max_iter = em3::EM3_INIT_GRID_ITER;
    const unsigned int rank_global = m_uiMesh->getMPIRankGlobal();
    MPI_Comm gcomm = m_uiMesh->getMPIGlobalCommunicator();

    DendroScalar* unzipVar[em3::EM3_NUM_VARS];
    unsigned int refineVarIds[em3::EM3_NUM_REFINE_VARS];

    for (unsigned int vIndex = 0; vIndex < em3::EM3_NUM_REFINE_VARS; vIndex++) {
        refineVarIds[vIndex] = em3::EM3_REFINE_VARIABLE_INDICES[vIndex];
    }

    double wTol = em3::EM3_WAVELET_TOL;
    std::function<double(double, double, double, double* hx)> waveletTolFunc =
        [](double x, double y, double z, double* hx) {
            return em3::computeWTol(x, y, z, hx);
        };

    DVec& m_evar = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ_IN];

    do {
        // initial unzipping step for stuff
        this->unzip(m_evar, m_evar_unz, em3::EM3_ASYNC_COMM_K);

        m_evar_unz.to_2d(unzipVar);

        if (em3::EM3_ENABLE_BLOCK_ADAPTIVITY)
            isRefine = false;
        else {
            // this is the remesh forcing based on the rk4 em3 refinement
            // option...
            isRefine = em3::isRemeshForce(
                m_uiMesh, (const double**)unzipVar, em3::VAR::U_B0,
                em3::EM3_CHI_REFINE_VAL, em3::EM3_CHI_COARSEN_VAL, true);
        }

        if (isRefine) {
            ot::Mesh* newMesh =
                this->remesh(em3::EM3_DENDRO_GRAIN_SZ, em3::EM3_LOAD_IMB_TOL,
                             em3::EM3_SPLIT_FIX);

            oldElements = m_uiMesh->getNumLocalMeshElements();
            newElements = newMesh->getNumLocalMeshElements();

            oldGridPoints = m_uiMesh->getNumLocalMeshNodes();
            newGridPoints = newMesh->getNumLocalMeshNodes();

            par::Mpi_Allreduce(&oldElements, &oldElements_g, 1, MPI_SUM, gcomm);
            par::Mpi_Allreduce(&newElements, &newElements_g, 1, MPI_SUM, gcomm);

            par::Mpi_Allreduce(&oldGridPoints, &oldGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());
            par::Mpi_Allreduce(&newGridPoints, &newGridPoints_g, 1, MPI_SUM,
                               m_uiMesh->getMPIGlobalCommunicator());

            if (!rank_global) {
                std::cout << GRN << "[EM3Ctx] init grid iter : " << iterCount
                          << "\t(Remesh Triggered!) -> Old Mesh : "
                          << oldElements_g << " New Mesh : " << newElements_g
                          << NRM << std::endl;
                std::cout << YLW << "                             "
                          << "\t                    -> Old Mesh (zip nodes): "
                          << oldGridPoints_g
                          << " New Mesh (zip nodes): " << newGridPoints_g << NRM
                          << std::endl;
            }

            this->grid_transfer(newMesh);

            std::swap(m_uiMesh, newMesh);
            delete newMesh;
        }

        iterCount += 1;
    } while (isRefine &&
             (newElements_g != oldElements_g ||
              newGridPoints_g != oldGridPoints_g) &&
             (iterCount < max_iter));

    this->init_grid();

    // reallocate space
    deallocate_em3_deriv_workspace();
    allocate_em3_deriv_workspace(m_uiMesh, 1);

    unsigned int lmin, lmax;
    m_uiMesh->computeMinMaxLevel(lmin, lmax);
    em3::EM3_RK45_TIME_STEP_SIZE =
        em3::EM3_CFL_FACTOR *
        ((em3::EM3_COMPD_MAX[0] - em3::EM3_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)em3::EM3_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    m_uiTinfo._m_uiTh = em3::EM3_RK45_TIME_STEP_SIZE;

    if (!m_uiMesh->getMPIRankGlobal()) {
        const DendroScalar dx_finest =
            ((em3::EM3_COMPD_MAX[0] - em3::EM3_COMPD_MIN[0]) *
             ((1u << (m_uiMaxDepth - lmax)) / ((double)em3::EM3_ELE_ORDER)) /
             ((double)(1u << (m_uiMaxDepth))));
        const DendroScalar dt_finest = em3::EM3_CFL_FACTOR * dx_finest;

        std::cout << BLU
                  << "================= Grid Info (After init grid "
                     "converge):==============================================="
                     "========"
                  << std::endl;
        std::cout << "lmin: " << lmin << " lmax:" << lmax << std::endl;
        std::cout << "dx: " << dx_finest << std::endl;
        std::cout << "dt: " << dt_finest << std::endl;
        std::cout << "========================================================="
                     "======================================================"
                  << NRM << std::endl;
    }

    // then initialize CFD
    // set up the actual compact finite difference object for the size
    unsigned int sz2 = 2 * em3::EM3_ELE_ORDER + 1;
    // NOTE: the
    dendro_cfd::cfd.set_filter_type(em3::EM3_FILTER_TYPE);
    dendro_cfd::cfd.set_deriv_type(em3::EM3_DERIV_TYPE);
    dendro_cfd::cfd.set_padding_size(em3::EM3_PADDING_WIDTH);
    // NOTE: the changing of dim size will reinitialize everything if the size
    // is different meaning that the main matrix will be recalculated and the
    // filter matrix will be recalculated
    dendro_cfd::cfd.change_dim_size(sz2);

    return 0;
}

int EM3Ctx::init_grid() {
    DVec& m_evar = m_var[VL::CPU_EV];
    // potential GPU code
    // DVec& m_dptr_evar

    const ot::TreeNode* pNodes = &(*(m_uiMesh->getAllElements().begin()));
    const unsigned int eleOrder = m_uiMesh->getElementOrder();
    const unsigned int* e2n_cg = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int* e2n_dg = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

    DendroScalar* zipIn[em3::EM3_NUM_VARS];
    m_evar.to_2d(zipIn);

    DendroScalar mp, mm, mp_adm, mm_adm, E, J1, J2, J3;

    for (unsigned int elem = m_uiMesh->getElementLocalBegin();
         elem < m_uiMesh->getElementLocalEnd(); elem++) {
        DendroScalar var[em3::EM3_NUM_VARS];
        for (unsigned int k = 0; k < (eleOrder + 1); k++)
            for (unsigned int j = 0; j < (eleOrder + 1); j++)
                for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                    const unsigned int nodeLookUp_CG =
                        e2n_cg[elem * nPe +
                               k * (eleOrder + 1) * (eleOrder + 1) +
                               j * (eleOrder + 1) + i];
                    if (nodeLookUp_CG >= nodeLocalBegin &&
                        nodeLookUp_CG < nodeLocalEnd) {
                        const unsigned int nodeLookUp_DG =
                            e2n_dg[elem * nPe +
                                   k * (eleOrder + 1) * (eleOrder + 1) +
                                   j * (eleOrder + 1) + i];
                        unsigned int ownerID, ii_x, jj_y, kk_z;
                        m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);
                        const DendroScalar len =
                            (double)(1u << (m_uiMaxDepth -
                                            pNodes[ownerID].getLevel()));

                        const DendroScalar x =
                            pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        const DendroScalar y =
                            pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        const DendroScalar z =
                            pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        em3::initData((double)x, (double)y, (double)z, var);

                        for (unsigned int v = 0; v < em3::EM3_NUM_VARS; v++)
                            zipIn[v][nodeLookUp_CG] = var[v];
                    }
                }
    }

    return 0;
}

int EM3Ctx::finalize() { return 0; }

int EM3Ctx::rhs_blk(const DendroScalar* in, DendroScalar* out, unsigned int dof,
                    unsigned int local_blk_id, DendroScalar blk_time) {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::RHS_BLK].start();
#endif

    // all the variables should be packed together.
    const ot::Block* blkList = m_uiMesh->getLocalBlockList().data();
    const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();
    assert(local_blk_id < numBlocks);

    const unsigned int blk = local_blk_id;
    DendroScalar** unzipIn = new DendroScalar*[dof];
    DendroScalar** unzipOut = new DendroScalar*[dof];

    unsigned int offset;
    double ptmin[3], ptmax[3];
    unsigned int lsz[3];
    unsigned int bflag;
    double dx, dy, dz;

    const Point pt_min(em3::EM3_COMPD_MIN[0], em3::EM3_COMPD_MIN[1],
                       em3::EM3_COMPD_MIN[2]);
    const Point pt_max(em3::EM3_COMPD_MAX[0], em3::EM3_COMPD_MAX[1],
                       em3::EM3_COMPD_MAX[2]);
    const unsigned int PW = em3::EM3_PADDING_WIDTH;

    offset = blkList[blk].getOffset();
    lsz[0] = blkList[blk].getAllocationSzX();
    lsz[1] = blkList[blk].getAllocationSzY();
    lsz[2] = blkList[blk].getAllocationSzZ();

    const unsigned int NN = lsz[0] * lsz[1] * lsz[2];

    for (unsigned int v = 0; v < dof; v++) {
        unzipIn[v] = (DendroScalar*)(in + v * NN);
        unzipOut[v] = (DendroScalar*)(out + v * NN);
    }

    bflag = blkList[blk].getBlkNodeFlag();

    dx = blkList[blk].computeDx(pt_min, pt_max);
    dy = blkList[blk].computeDy(pt_min, pt_max);
    dz = blkList[blk].computeDz(pt_min, pt_max);

    ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
    ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
    ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

    ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
    ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
    ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

    // note that the offset zero is important since it is the block vector.
    em3rhs(unzipOut, (const DendroScalar**)unzipIn, 0, ptmin, ptmax, lsz,
           bflag);

    // for(unsigned int v =0; v < dof; v++)
    // {
    //     unsigned int nid=0;
    //     for(unsigned int k=3; k < lsz[2]-3; k++)
    //     for(unsigned int j=3; j < lsz[1]-3; j++)
    //     for(unsigned int i=3; i < lsz[0]-3; i++,nid++)
    //     {
    //         std::cout<<" blk::v: "<<v<<" n: "<<nid<< " val:
    //         "<<unzipOut[v][k*lsz[1]*lsz[0] +j*lsz[0] +i ]<<std::endl;;
    //     }
    // }

    delete[] unzipIn;
    delete[] unzipOut;

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::RHS_BLK].stop();
#endif

    return 0;
}

int EM3Ctx::write_vtu() {
    if (!m_uiMesh->isActive()) return 0;

    DVec& m_evar = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ_IN];
    DVec& m_cvar = m_var[VL::CPU_CV];
    DVec& m_cvar_unz = m_var[VL::CPU_CV_UZ_IN];
    DVec& m_pvar = m_var[VL::CPU_PV];

    // Make sure the primitives and constraints are calculated first!
    this->compute_primitives();  // pvars are exhanged when computing the prims.
    this->compute_constraints();  // evars and cvars are commuicated when
                                  // computing the constraints.


    std::vector<std::string> pDataNames;
    const unsigned int numConstVars = EM3_NUM_CONST_VARS_VTU_OUTPUT;
    const unsigned int numEvolVars = EM3_NUM_EVOL_VARS_VTU_OUTPUT;
    const unsigned int numPrimVars = EM3_NUM_EVOL_VARS_VTU_OUTPUT;

    DendroScalar* consVar[EM3_CONSTRAINT_NUM_VARS];
    DendroScalar* evolVar[EM3_NUM_VARS];
    DendroScalar* primVar[EM3_NUM_VARS];

    // make them to 2D
    m_evar.to_2d(evolVar);
    m_pvar.to_2d(primVar);
    m_cvar.to_2d(consVar);

    double* pData[(numConstVars + numEvolVars + numPrimVars)];

    for (unsigned int i = 0; i < numEvolVars; i++) {
        pDataNames.push_back(
            std::string(EM3_VAR_NAMES[EM3_VTU_OUTPUT_EVOL_INDICES[i]]));
        pData[i] = evolVar[EM3_VTU_OUTPUT_EVOL_INDICES[i]];
    }

    for (unsigned int i = 0; i < numConstVars; i++) {
        pDataNames.push_back(std::string(
            EM3_CONSTRAINT_VAR_NAMES[EM3_VTU_OUTPUT_CONST_INDICES[i]]));
        pData[numEvolVars + i] = consVar[EM3_VTU_OUTPUT_CONST_INDICES[i]];
    }

    for (unsigned int i = 0; i < numPrimVars; i++) {
        pDataNames.push_back(
            std::string("diff_") +
            std::string(EM3_VAR_NAMES[EM3_VTU_OUTPUT_EVOL_INDICES[i]]));
        pData[numConstVars + numEvolVars + i] =
            primVar[EM3_VTU_OUTPUT_EVOL_INDICES[i]];
    }

    std::vector<char*> pDataNames_char;
    pDataNames_char.reserve(pDataNames.size());

    for (unsigned int i = 0; i < pDataNames.size(); i++)
        pDataNames_char.push_back(const_cast<char*>(pDataNames[i].c_str()));

    const char* fDataNames[] = {"Time", "Cycle"};
    const double fData[] = {m_uiTinfo._m_uiT, (double)m_uiTinfo._m_uiStep};

    char fPrefix[256];
    sprintf(fPrefix, "%s_%d", EM3_VTU_FILE_PREFIX.c_str(), m_uiTinfo._m_uiStep);

    if (em3::EM3_VTU_Z_SLICE_ONLY) {
        unsigned int s_val[3] = {1u << (m_uiMaxDepth - 1),
                                 1u << (m_uiMaxDepth - 1),
                                 1u << (m_uiMaxDepth - 1)};
        unsigned int s_norm[3] = {0, 0, 1};
        io::vtk::mesh2vtu_slice(
            m_uiMesh, s_val, s_norm, fPrefix, 2, fDataNames, fData,
            (numEvolVars + numConstVars + numPrimVars),
            (const char**)&pDataNames_char[0], (const double**)pData);
    } else {
        io::vtk::mesh2vtuFine(m_uiMesh, fPrefix, 2, fDataNames, fData,
                              (numConstVars + numEvolVars + numPrimVars),
                              (const char**)&pDataNames_char[0],
                              (const double**)pData);
    }

    std::cout << "ending vtu out..." << std::endl;
    return 0;
}

int EM3Ctx::write_checkpt() {
    if (m_uiMesh->isActive()) {
        unsigned int cpIndex;
        (m_uiTinfo._m_uiStep % (2 * EM3_CHECKPT_FREQ) == 0)
            ? cpIndex = 0
            : cpIndex = 1;  // to support alternate file writing.
        unsigned int rank = m_uiMesh->getMPIRank();
        unsigned int npes = m_uiMesh->getMPICommSize();

        char fName[256];
        const ot::TreeNode* pNodes = &(*(m_uiMesh->getAllElements().begin() +
                                         m_uiMesh->getElementLocalBegin()));
        sprintf(fName, "%s_octree_%d_%d.oct", EM3_CHKPT_FILE_PREFIX.c_str(),
                cpIndex, rank);
        io::checkpoint::writeOctToFile(fName, pNodes,
                                       m_uiMesh->getNumLocalMeshElements());

        unsigned int numVars = EM3_NUM_VARS;
        const char** varNames = EM3_VAR_NAMES;

        DVec& m_evar = m_var[VL::CPU_EV];

        DendroScalar* eVar[EM3_NUM_VARS];
        m_evar.to_2d(eVar);

        sprintf(fName, "%s_%d_%d.var", EM3_CHKPT_FILE_PREFIX.c_str(), cpIndex,
                rank);
        io::checkpoint::writeVecToFile(fName, m_uiMesh, (const double**)eVar,
                                       em3::EM3_NUM_VARS);

        if (!rank) {
            sprintf(fName, "%s_step_%d.cp", EM3_CHKPT_FILE_PREFIX.c_str(),
                    cpIndex);
            // std::cout<<"writing : "<<fName<<std::endl;
            std::ofstream outfile(fName);
            if (!outfile) {
                std::cout << fName << " file open failed " << std::endl;
                return 0;
            }

            json checkPoint;

            checkPoint["DENDRO_TS_TIME_BEGIN"] = m_uiTinfo._m_uiTb;
            checkPoint["DENDRO_TS_TIME_END"] = m_uiTinfo._m_uiTe;
            checkPoint["DENDRO_TS_ELEMENT_ORDER"] = m_uiElementOrder;

            checkPoint["DENDRO_TS_TIME_CURRENT"] = m_uiTinfo._m_uiT;
            checkPoint["DENDRO_TS_STEP_CURRENT"] = m_uiTinfo._m_uiStep;
            checkPoint["DENDRO_TS_TIME_STEP_SIZE"] = m_uiTinfo._m_uiTh;
            checkPoint["DENDRO_TS_LAST_IO_TIME"] = m_uiTinfo._m_uiT;

            checkPoint["DENDRO_TS_WAVELET_TOLERANCE"] = EM3_WAVELET_TOL;
            checkPoint["DENDRO_TS_LOAD_IMB_TOLERANCE"] = EM3_LOAD_IMB_TOL;
            checkPoint["DENDRO_TS_NUM_VARS"] =
                numVars;  // number of variables to restore.
            checkPoint["DENDRO_TS_ACTIVE_COMM_SZ"] =
                m_uiMesh
                    ->getMPICommSize();  // (note that rank 0 is always active).

            outfile << std::setw(4) << checkPoint << std::endl;
            outfile.close();
        }
    }

    return 0;
}

int EM3Ctx::restore_checkpt() {
    unsigned int numVars = 0;
    std::vector<ot::TreeNode> octree;
    json checkPoint;

    int rank;
    int npes;
    MPI_Comm comm = m_uiMesh->getMPIGlobalCommunicator();
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    unsigned int activeCommSz;

    char fName[256];
    unsigned int restoreStatus = 0;
    unsigned int restoreStatusGlobal =
        0;  // 0 indicates successfully restorable.

    ot::Mesh* newMesh;
    unsigned int restoreStep[2];
    restoreStep[0] = 0;
    restoreStep[1] = 0;

    unsigned int restoreFileIndex = 0;

    for (unsigned int cpIndex = 0; cpIndex < 2; cpIndex++) {
        restoreStatus = 0;

        if (!rank) {
            sprintf(fName, "%s_step_%d.cp", EM3_CHKPT_FILE_PREFIX.c_str(),
                    cpIndex);
            std::ifstream infile(fName);
            if (!infile) {
                std::cout << fName << " file open failed " << std::endl;
                restoreStatus = 1;
            }

            if (restoreStatus == 0) {
                infile >> checkPoint;
                m_uiTinfo._m_uiTb = checkPoint["DENDRO_TS_TIME_BEGIN"];
                m_uiTinfo._m_uiTe = checkPoint["DENDRO_TS_TIME_END"];
                m_uiTinfo._m_uiT = checkPoint["DENDRO_TS_TIME_CURRENT"];
                m_uiTinfo._m_uiStep = checkPoint["DENDRO_TS_STEP_CURRENT"];
                m_uiTinfo._m_uiTh = checkPoint["DENDRO_TS_TIME_STEP_SIZE"];
                m_uiElementOrder = checkPoint["DENDRO_TS_ELEMENT_ORDER"];

                EM3_WAVELET_TOL = checkPoint["DENDRO_TS_WAVELET_TOLERANCE"];
                EM3_LOAD_IMB_TOL = checkPoint["DENDRO_TS_LOAD_IMB_TOLERANCE"];

                numVars = checkPoint["DENDRO_TS_NUM_VARS"];
                activeCommSz = checkPoint["DENDRO_TS_ACTIVE_COMM_SZ"];

                restoreStep[cpIndex] = m_uiTinfo._m_uiStep;
            }
        }
    }

    if (!rank) {
        if (restoreStep[0] < restoreStep[1])
            restoreFileIndex = 1;
        else
            restoreFileIndex = 0;
    }

    par::Mpi_Bcast(&restoreFileIndex, 1, 0, comm);

    restoreStatus = 0;
    octree.clear();
    if (!rank)
        std::cout << "[EM3Ctx] :  Trying to restore from checkpoint index : "
                  << restoreFileIndex << std::endl;

    if (!rank) {
        sprintf(fName, "%s_step_%d.cp", EM3_CHKPT_FILE_PREFIX.c_str(),
                restoreFileIndex);
        std::ifstream infile(fName);
        if (!infile) {
            std::cout << fName << " file open failed " << std::endl;
            restoreStatus = 1;
        }

        if (restoreStatus == 0) {
            infile >> checkPoint;
            m_uiTinfo._m_uiTb = checkPoint["DENDRO_TS_TIME_BEGIN"];
            m_uiTinfo._m_uiTe = checkPoint["DENDRO_TS_TIME_END"];
            m_uiTinfo._m_uiT = checkPoint["DENDRO_TS_TIME_CURRENT"];
            m_uiTinfo._m_uiStep = checkPoint["DENDRO_TS_STEP_CURRENT"];
            m_uiTinfo._m_uiTh = checkPoint["DENDRO_TS_TIME_STEP_SIZE"];
            m_uiElementOrder = checkPoint["DENDRO_TS_ELEMENT_ORDER"];

            EM3_WAVELET_TOL = checkPoint["DENDRO_TS_WAVELET_TOLERANCE"];
            EM3_LOAD_IMB_TOL = checkPoint["DENDRO_TS_LOAD_IMB_TOLERANCE"];

            numVars = checkPoint["DENDRO_TS_NUM_VARS"];
            activeCommSz = checkPoint["DENDRO_TS_ACTIVE_COMM_SZ"];

            restoreStep[restoreFileIndex] = m_uiTinfo._m_uiStep;
        }
    }

    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout
                << "[EM3Ctx] : Restore step failed, restore file corrupted. "
                << std::endl;
        MPI_Abort(comm, 0);
    }

    MPI_Bcast(&m_uiTinfo, sizeof(ts::TSInfo), MPI_BYTE, 0, comm);
    par::Mpi_Bcast(&EM3_WAVELET_TOL, 1, 0, comm);
    par::Mpi_Bcast(&EM3_LOAD_IMB_TOL, 1, 0, comm);

    par::Mpi_Bcast(&numVars, 1, 0, comm);
    par::Mpi_Bcast(&m_uiElementOrder, 1, 0, comm);
    par::Mpi_Bcast(&activeCommSz, 1, 0, comm);

    if (activeCommSz > npes) {
        if (!rank)
            std::cout
                << " [EM3Ctx] : checkpoint file written from  a larger "
                   "communicator than the current global comm. (i.e. "
                   "communicator shrinking not allowed in the restore step. )"
                << std::endl;

        MPI_Abort(comm, 0);
    }

    bool isActive = (rank < activeCommSz);

    MPI_Comm newComm;
    par::splitComm2way(isActive, &newComm, comm);

    if (isActive) {
        int activeRank;
        int activeNpes;

        MPI_Comm_rank(newComm, &activeRank);
        MPI_Comm_size(newComm, &activeNpes);
        assert(activeNpes == activeCommSz);

        sprintf(fName, "%s_octree_%d_%d.oct", EM3_CHKPT_FILE_PREFIX.c_str(),
                restoreFileIndex, activeRank);
        restoreStatus = io::checkpoint::readOctFromFile(fName, octree);
        assert(par::test::isUniqueAndSorted(octree, newComm));
    }

    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout << "[EM3Ctx]: octree (*.oct) restore file is corrupted "
                      << std::endl;
        MPI_Abort(comm, 0);
    }

    newMesh = new ot::Mesh(octree, 1, m_uiElementOrder, activeCommSz, comm);
    newMesh->setDomainBounds(
        Point(em3::EM3_GRID_MIN_X, em3::EM3_GRID_MIN_Y, em3::EM3_GRID_MIN_Z),
        Point(em3::EM3_GRID_MAX_X, em3::EM3_GRID_MAX_Y, em3::EM3_GRID_MAX_Z));
    // no need to transfer data only to resize the contex variables.
    // no need to transfer data only to resize the contex variables.
    // this->grid_transfer(newMesh);
    for (unsigned int i = 0; i < VL::END; i++) m_var[i].destroy_vector();

    m_var[VL::CPU_EV].create_vector(newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, EM3_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_OUT].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_NUM_VARS, true);

    m_var[VL::CPU_CV].create_vector(newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, EM3_CONSTRAINT_NUM_VARS,
                                    true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        newMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_CONSTRAINT_NUM_VARS, true);

    m_var[VL::CPU_PV].create_vector(newMesh, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, EM3_NUM_VARS, true);

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, EM3_NUM_VARS,
                                      EM3_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(newMesh, m_mpi_ctx, EM3_NUM_VARS,
                                    EM3_ASYNC_COMM_K);

    // only reads the evolution variables.
    if (isActive) {
        int activeRank;
        int activeNpes;

        DendroScalar* inVec[EM3_NUM_VARS];
        DVec& m_evar = m_var[VL::CPU_EV];
        m_evar.to_2d(inVec);

        MPI_Comm_rank(newComm, &activeRank);
        MPI_Comm_size(newComm, &activeNpes);
        assert(activeNpes == activeCommSz);

        sprintf(fName, "%s_%d_%d.var", EM3_CHKPT_FILE_PREFIX.c_str(),
                restoreFileIndex, activeRank);
        restoreStatus = io::checkpoint::readVecFromFile(fName, newMesh, inVec,
                                                        EM3_NUM_VARS);
    }

    MPI_Comm_free(&newComm);
    par::Mpi_Allreduce(&restoreStatus, &restoreStatusGlobal, 1, MPI_MAX, comm);
    if (restoreStatusGlobal == 1) {
        if (!rank)
            std::cout << "[EM3Ctx]: varible (*.var) restore file currupted "
                      << std::endl;
        MPI_Abort(comm, 0);
    }

    std::swap(m_uiMesh, newMesh);
    delete newMesh;

    // realloc em3 deriv space
    deallocate_em3_deriv_workspace();
    allocate_em3_deriv_workspace(m_uiMesh, 1);

    unsigned int localSz = m_uiMesh->getNumLocalMeshElements();
    unsigned int totalElems = 0;
    par::Mpi_Allreduce(&localSz, &totalElems, 1, MPI_SUM, comm);

    if (!rank)
        std::cout << " checkpoint at step : " << m_uiTinfo._m_uiStep
                  << "active Comm. sz: " << activeCommSz
                  << " restore successful: "
                  << " restored mesh size: " << totalElems << std::endl;

    m_uiIsETSSynced = false;
    return 0;
}

// int EM3Ctx::pre_timestep(DVec sIn) { return 0; }

// int EM3Ctx::pre_stage(DVec sIn) { return 0; }

int EM3Ctx::post_stage(DVec& sIn) { return 0; }

int EM3Ctx::post_timestep(DVec& sIn) { return 0; }

bool EM3Ctx::is_remesh() {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::IS_REMESH].start();
#endif

    bool isRefine = false;

    if (EM3_ENABLE_BLOCK_ADAPTIVITY) return false;

    MPI_Comm comm = m_uiMesh->getMPIGlobalCommunicator();

    DVec& m_evar = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ_IN];

    this->unzip(m_evar, m_evar_unz, EM3_ASYNC_COMM_K);

    DendroScalar* unzipVar[EM3_NUM_VARS];
    m_evar_unz.to_2d(unzipVar);

    unsigned int refineVarIds[EM3_NUM_REFINE_VARS];
    for (unsigned int vIndex = 0; vIndex < EM3_NUM_REFINE_VARS; vIndex++)
        refineVarIds[vIndex] = EM3_REFINE_VARIABLE_INDICES[vIndex];

    double wTol = EM3_WAVELET_TOL;
    std::function<double(double, double, double, double*)> waveletTolFunc =
        [wTol](double x, double y, double z, double* hx) {
            return computeWTol(x, y, z, hx);
        };

    if (em3::EM3_REFINEMENT_MODE == em3::RefinementMode::WAMR)
        isRefine = m_uiMesh->isReMeshUnzip(
            (const double**)unzipVar, refineVarIds, EM3_NUM_REFINE_VARS,
            waveletTolFunc, em3::EM3_DENDRO_AMR_FAC);

    if (em3::EM3_REFINEMENT_MODE == em3::RefinementMode::FR)
        isRefine = em3::isRemeshForce(m_uiMesh, (const double**)unzipVar,
                                      em3::VAR::U_B0, em3::EM3_CHI_REFINE_VAL,
                                      em3::EM3_CHI_COARSEN_VAL, true);

    if (em3::EM3_REFINEMENT_MODE == em3::RefinementMode::WAMR_FR) {
        const bool isRefine1 = m_uiMesh->isReMeshUnzip(
            (const double**)unzipVar, refineVarIds, EM3_NUM_REFINE_VARS,
            waveletTolFunc, em3::EM3_DENDRO_AMR_FAC);
        const bool isRefine2 = isRefine = em3::isRemeshForce(
            m_uiMesh, (const double**)unzipVar, em3::VAR::U_B0,
            em3::EM3_CHI_REFINE_VAL, em3::EM3_CHI_COARSEN_VAL, false);
        isRefine = (isRefine1 || isRefine2);
    }

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::IS_REMESH].stop();
#endif

    return isRefine;
}

DVec& EM3Ctx::get_evolution_vars() { return m_var[CPU_EV]; }

DVec& EM3Ctx::get_constraint_vars() { return m_var[CPU_CV]; }

DVec& EM3Ctx::get_primitive_vars() { return m_var[CPU_PV]; }

int EM3Ctx::compute_constraints() {
    if (!m_uiMesh->isActive()) return 0;

    if (!m_uiMesh->getMPIRank()) {
        std::cout << "... Now computing constraints" << std::endl;
    }

        std::cout << "... Now computing constraints (each proc)" << std::endl;

    DVec& m_evar = m_var[VL::CPU_EV];
    DVec& m_evar_unz = m_var[VL::CPU_EV_UZ_IN];
    DVec& m_cvar = m_var[VL::CPU_CV];
    DVec& m_cvar_unz = m_var[VL::CPU_CV_UZ_IN];
    DVec& m_pvar = m_var[VL::CPU_PV];

    // unzip the evolution variables
    this->unzip(m_evar, m_evar_unz, EM3_ASYNC_COMM_K);

    DendroScalar* consVar[EM3_CONSTRAINT_NUM_VARS];
    DendroScalar* consVarUnzip[EM3_CONSTRAINT_NUM_VARS];
    DendroScalar* evolVar[EM3_NUM_VARS];
    DendroScalar* evolVarUnzip[EM3_NUM_VARS];
    DendroScalar* primVar[EM3_NUM_VARS];

    // make everything 2D
    m_evar.to_2d(evolVar);
    m_pvar.to_2d(primVar);
    m_cvar.to_2d(consVar);
    m_evar_unz.to_2d(evolVarUnzip);
    m_cvar_unz.to_2d(consVarUnzip);

    // now we can go through the blocks
    const std::vector<ot::Block> blkList = m_uiMesh->getLocalBlockList();
    unsigned int offset;
    double ptmin[3], ptmax[3];
    unsigned int sz[3];
    unsigned int bflag;
    double dx, dy, dz;
    const Point pt_min(em3::EM3_COMPD_MIN[0], em3::EM3_COMPD_MIN[1],
                       em3::EM3_COMPD_MIN[2]);
    const Point pt_max(em3::EM3_COMPD_MAX[0], em3::EM3_COMPD_MAX[1],
                       em3::EM3_COMPD_MAX[2]);
    const unsigned int PW = em3::EM3_PADDING_WIDTH;

    for (unsigned int blk = 0; blk < blkList.size(); blk++) {
        offset = blkList[blk].getOffset();
        sz[0] = blkList[blk].getAllocationSzX();
        sz[1] = blkList[blk].getAllocationSzY();
        sz[2] = blkList[blk].getAllocationSzZ();

        bflag = blkList[blk].getBlkNodeFlag();

        dx = blkList[blk].computeDx(pt_min, pt_max);
        dy = blkList[blk].computeDy(pt_min, pt_max);
        dz = blkList[blk].computeDz(pt_min, pt_max);

        ptmin[0] = GRIDX_TO_X(blkList[blk].getBlockNode().minX()) - PW * dx;
        ptmin[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().minY()) - PW * dy;
        ptmin[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ()) - PW * dz;

        ptmax[0] = GRIDX_TO_X(blkList[blk].getBlockNode().maxX()) + PW * dx;
        ptmax[1] = GRIDY_TO_Y(blkList[blk].getBlockNode().maxY()) + PW * dy;
        ptmax[2] = GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ()) + PW * dz;

        physical_constraints(consVarUnzip, (const DendroScalar**)evolVarUnzip,
                             offset, ptmin, ptmax, sz, bflag);
    }

    // zip up the computed physical constraints...
    this->zip(m_cvar_unz, m_cvar);

    // do the ghost exchanging
    m_uiMesh->readFromGhostBegin(m_cvar.get_vec_ptr(), m_cvar.get_dof());
    m_uiMesh->readFromGhostEnd(m_cvar.get_vec_ptr(), m_cvar.get_dof());

    // now we can compute the L2 Norm information...

    DendroScalar l2_norm[EM3_CONSTRAINT_NUM_VARS];
    DendroScalar l2_rs[EM3_CONSTRAINT_NUM_VARS];
    DendroScalar vmin[EM3_CONSTRAINT_NUM_VARS];
    DendroScalar vmax[EM3_CONSTRAINT_NUM_VARS];

    for (unsigned int v = 0; v < EM3_CONSTRAINT_NUM_VARS; v++) {
        l2_norm[v] = normL2(m_uiMesh, consVar[v], ot::VEC_TYPE::CG_NODAL, true);
        l2_rs[v] = rsNormLp(m_uiMesh, consVar[v], 2);
        vmin[v] = vecMin(m_uiMesh, consVar[v], ot::VEC_TYPE::CG_NODAL, true);
        vmax[v] = vecMax(m_uiMesh, consVar[v], ot::VEC_TYPE::CG_NODAL, true);
    }

    // TODO: extract constraints

    return 0;
}

int EM3Ctx::compute_primitives() {
    if (!m_uiMesh->isActive()) return 0;

    DendroScalar* consVar[EM3_CONSTRAINT_NUM_VARS];
    DendroScalar* evolVar[EM3_NUM_VARS];
    DendroScalar* primVar[EM3_NUM_VARS];

    DVec& m_evar = m_var[VL::CPU_EV];
    DVec& m_cvar = m_var[VL::CPU_CV];
    DVec& m_pvar = m_var[VL::CPU_PV];

    m_evar.to_2d(evolVar);
    m_cvar.to_2d(consVar);
    m_pvar.to_2d(primVar);

    if (!m_uiMesh->getMPIRank()) {
        std::cout << "... Now computing difference" << std::endl;
    }

    // initialize diff begin.
    unsigned int nodeLookUp_CG;
    unsigned int nodeLookUp_DG;
    double x, y, z, len;
    const ot::TreeNode* pNodes = &(*(m_uiMesh->getAllElements().begin()));
    unsigned int ownerID, ii_x, jj_y, kk_z;
    unsigned int eleOrder = m_uiMesh->getElementOrder();
    const unsigned int* e2n_cg = &(*(m_uiMesh->getE2NMapping().begin()));
    const unsigned int* e2n_dg = &(*(m_uiMesh->getE2NMapping_DG().begin()));
    const unsigned int nPe = m_uiMesh->getNumNodesPerElement();
    const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
    const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

    double var[em3::EM3_NUM_VARS];
    for (unsigned int elem = m_uiMesh->getElementLocalBegin();
         elem < m_uiMesh->getElementLocalEnd(); elem++) {
        for (unsigned int k = 0; k < (eleOrder + 1); k++)
            for (unsigned int j = 0; j < (eleOrder + 1); j++)
                for (unsigned int i = 0; i < (eleOrder + 1); i++) {
                    nodeLookUp_CG = e2n_cg[elem * nPe +
                                           k * (eleOrder + 1) * (eleOrder + 1) +
                                           j * (eleOrder + 1) + i];
                    if (nodeLookUp_CG >= nodeLocalBegin &&
                        nodeLookUp_CG < nodeLocalEnd) {
                        nodeLookUp_DG =
                            e2n_dg[elem * nPe +
                                   k * (eleOrder + 1) * (eleOrder + 1) +
                                   j * (eleOrder + 1) + i];
                        m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y,
                                          kk_z);
                        len = (double)(1u << (m_uiMaxDepth -
                                              pNodes[ownerID].getLevel()));
                        x = pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                        y = pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                        z = pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                        em3::analyticalSol((double)x, (double)y, (double)z,
                                           m_uiTinfo._m_uiT, var);
                        for (unsigned int v = 0; v < em3::EM3_NUM_VARS; v++)
                            primVar[v][nodeLookUp_CG] =
                                evolVar[v][nodeLookUp_CG] - var[v];
                    }
                }
    }

    // pass the information along from zipped vector...
    m_uiMesh->readFromGhostBegin(m_pvar.get_vec_ptr(), m_pvar.get_dof());
    m_uiMesh->readFromGhostEnd(m_pvar.get_vec_ptr(), m_pvar.get_dof());

    // TODO: any sort of extraction?

    return 0;
}

int EM3Ctx::terminal_output() {
    if (m_uiMesh->isActive()) {
        const unsigned int currentStep = m_uiTinfo._m_uiStep;

        DVec& m_evar = m_var[VL::CPU_EV];
        DVec& m_cvar = m_var[VL::CPU_CV];
        DVec& m_pvar = m_var[VL::CPU_PV];

            // compare with the analytical solution.
            DendroScalar* consVar[EM3_CONSTRAINT_NUM_VARS];
            DendroScalar* evolVar[EM3_NUM_VARS];
            DendroScalar* primVar[EM3_NUM_VARS];

            m_evar.to_2d(evolVar);
            m_pvar.to_2d(primVar);
            m_cvar.to_2d(consVar);

            // this->compute_primitives();

            DendroScalar l2_norm[EM3_NUM_VARS];
            DendroScalar l2_rs[EM3_NUM_VARS];
            DendroScalar vmin[EM3_NUM_VARS];
            DendroScalar vmax[EM3_NUM_VARS];

            for (unsigned int v = 0; v < EM3_NUM_VARS; v++) {
                l2_norm[v] =
                    normL2(m_uiMesh, primVar[v], ot::VEC_TYPE::CG_NODAL, true);
                l2_rs[v] = rsNormLp(m_uiMesh, primVar[v], 2);
                vmin[v] =
                    vecMin(m_uiMesh, primVar[v], ot::VEC_TYPE::CG_NODAL, true);
                vmax[v] =
                    vecMax(m_uiMesh, primVar[v], ot::VEC_TYPE::CG_NODAL, true);
            }

            if (!m_uiMesh->getMPIRank()) {
                std::cout << RED << "======== SOLVER UPDATE ========" << NRM
                          << std::endl;
                std::cout << BLU << "Executing step: " << m_uiTinfo._m_uiStep
                          << " dt: " << m_uiTinfo._m_uiTh
                          << " rk_time : " << m_uiTinfo._m_uiT << std::endl;
                std::cout << GRN << "    == Evolution Variables == " << NRM
                          << std::endl;
            }

            for (unsigned int v = 0; v < EM3_NUM_VARS; v++) {
                DendroScalar min = 0, max = 0;
                min =
                    vecMin(m_uiMesh, evolVar[v], ot::VEC_TYPE::CG_NODAL, true);
                max =
                    vecMax(m_uiMesh, evolVar[v], ot::VEC_TYPE::CG_NODAL, true);
                if (!(m_uiMesh->getMPIRank()))
                    std::cout << "\t ||" << EM3_VAR_NAMES[v]
                              << "|| (min, max) : \t ( " << min << ", " << max
                              << " ) " << std::endl;
            }

            if (!m_uiMesh->getMPIRank()) {
                std::cout << GRN << "    == Constraint Variables == " << NRM
                          << std::endl;

                for (unsigned int v = 0; v < EM3_CONSTRAINT_NUM_VARS; v++)
                    std::cout << "\t ||" << EM3_CONSTRAINT_VAR_NAMES[v]
                              << "|| (min, max, l2, l2rs) : (" << vmin[v]
                              << ", " << vmax[v] << ", " << l2_norm[v] << ", "
                              << l2_rs[v] << " ) " << std::endl;
            }

            if (!m_uiMesh->getMPIRank()) {
                std::cout << GRN
                          << "    == Difference With Analytical Solution == "
                          << NRM << std::endl;

                for (unsigned int v = 0; v < EM3_NUM_VARS; v++)
                    std::cout << "\t ||diff_[" << EM3_VAR_NAMES[v]
                              << "]|| (min, max, l2, l2rs) : (" << vmin[v]
                              << ", " << vmax[v] << ", " << l2_norm[v] << ", "
                              << l2_rs[v] << " ) " << std::endl;
            }
    }

    return 0;
}

int EM3Ctx::grid_transfer(const ot::Mesh* m_new) {
#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].start();
#endif
    DVec& m_evar = m_var[VL::CPU_EV];
    DVec::grid_transfer(m_uiMesh, m_new, m_evar);
    // printf("igt ended\n");

    m_var[VL::CPU_CV].destroy_vector();
    m_var[VL::CPU_CV_UZ_IN].destroy_vector();

    m_var[VL::CPU_EV_UZ_IN].destroy_vector();
    m_var[VL::CPU_EV_UZ_OUT].destroy_vector();

    m_var[VL::CPU_PV].destroy_vector();

    m_var[VL::CPU_CV].create_vector(m_new, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, EM3_CONSTRAINT_NUM_VARS,
                                    true);
    m_var[VL::CPU_CV_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_CONSTRAINT_NUM_VARS, true);

    m_var[VL::CPU_EV_UZ_IN].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_NUM_VARS, true);
    m_var[VL::CPU_EV_UZ_OUT].create_vector(
        m_new, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,
        EM3_NUM_VARS, true);

    m_var[VL::CPU_PV].create_vector(m_new, ot::DVEC_TYPE::OCT_SHARED_NODES,
                                    ot::DVEC_LOC::HOST, EM3_NUM_VARS, true);

    ot::dealloc_mpi_ctx<DendroScalar>(m_uiMesh, m_mpi_ctx, EM3_NUM_VARS,
                                      EM3_ASYNC_COMM_K);
    ot::alloc_mpi_ctx<DendroScalar>(m_new, m_mpi_ctx, EM3_NUM_VARS,
                                    EM3_ASYNC_COMM_K);

    m_uiIsETSSynced = false;

#ifdef __PROFILE_CTX__
    m_uiCtxpt[ts::CTXPROFILE::GRID_TRASFER].stop();
#endif
    return 0;
}

unsigned int EM3Ctx::getBlkTimestepFac(unsigned int blev, unsigned int lmin,
                                       unsigned int lmax) {
    const unsigned int ldiff = 0;
    if ((lmax - blev) <= ldiff)
        return 1;
    else {
        return 1u << (lmax - blev - ldiff);
    }
}

}  // end of namespace em3