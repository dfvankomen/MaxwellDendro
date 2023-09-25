#include <iostream>
#include <vector>

#include "TreeNode.h"
#include "em3.h"
#include "em3Ctx.h"
#include "mesh.h"
#include "meshUtils.h"
#include "mpi.h"
#include "rk4em3.h"
#include "sdc.h"
#include "ets.h"
#include "parameters.h"

int main(int argc, char** argv) {
    unsigned int ts_mode = 1;

    if (argc < 2) {
        std::cout << "Usage: " << argv[0]
                  << " paramFile TSMode(1){0-Spatially Adaptive Time Stepping "
                     "(SATS), 1- Uniform Time Stepping ("
                  << GRN << "default" << NRM << ")}" << std::endl;
        return 0;
    }

    if (argc > 2) ts_mode = std::atoi(argv[2]);

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, npes;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);

    // 1 . read the parameter file.
    if (!rank) std::cout << " reading parameter file :" << argv[1] << std::endl;
    em3::readParamFile(argv[1], comm);

    if (rank == 1 || npes == 1) {
        std::cout << "parameters read: " << std::endl;

        std::cout << YLW << "\tnpes :" << npes << NRM << std::endl;
        std::cout << YLW << "\tEM3_ELE_ORDER :" << em3::EM3_ELE_ORDER << NRM
                  << std::endl;
        std::cout << YLW << "\tEM3_PADDING_WIDTH :" << em3::EM3_PADDING_WIDTH
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_DIM :" << em3::EM3_DIM << NRM << std::endl;
        std::cout << YLW << "\tEM3_IO_OUTPUT_FREQ :" << em3::EM3_IO_OUTPUT_FREQ
                  << NRM << std::endl;
        std::cout << YLW
                  << "\tEM3_REMESH_TEST_FREQ :" << em3::EM3_REMESH_TEST_FREQ
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_CHECKPT_FREQ :" << em3::EM3_CHECKPT_FREQ
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_RESTORE_SOLVER :" << em3::EM3_RESTORE_SOLVER
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_ENABLE_BLOCK_ADAPTIVITY :"
                  << em3::EM3_ENABLE_BLOCK_ADAPTIVITY << NRM << std::endl;
        std::cout << YLW
                  << "\tEM3_VTU_FILE_PREFIX :" << em3::EM3_VTU_FILE_PREFIX
                  << NRM << std::endl;
        std::cout << YLW
                  << "\tEM3_CHKPT_FILE_PREFIX :" << em3::EM3_CHKPT_FILE_PREFIX
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_PROFILE_FILE_PREFIX :"
                  << em3::EM3_PROFILE_FILE_PREFIX << NRM << std::endl;
        std::cout << YLW
                  << "\tEM3_VTU_Z_SLICE_ONLY :" << em3::EM3_VTU_Z_SLICE_ONLY
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_IO_OUTPUT_GAP :" << em3::EM3_IO_OUTPUT_GAP
                  << NRM << std::endl;
        std::cout << YLW
                  << "\tEM3_DENDRO_GRAIN_SZ :" << em3::EM3_DENDRO_GRAIN_SZ
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_ASYNC_COMM_K :" << em3::EM3_ASYNC_COMM_K
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_DENDRO_AMR_FAC :" << em3::EM3_DENDRO_AMR_FAC
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_CFL_FACTOR:" << em3::EM3_CFL_FACTOR << NRM
                  << std::endl;
        std::cout << YLW << "\tEM3_WAVELET_TOL :" << em3::EM3_WAVELET_TOL << NRM
                  << std::endl;
        std::cout << YLW << "\tEM3_LOAD_IMB_TOL :" << em3::EM3_LOAD_IMB_TOL
                  << NRM << std::endl;
        std::cout << YLW
                  << "\tEM3_RK45_TIME_BEGIN :" << em3::EM3_RK45_TIME_BEGIN
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_RK45_TIME_END :" << em3::EM3_RK45_TIME_END
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_RK45_TIME_STEP_SIZE :"
                  << em3::EM3_RK45_TIME_STEP_SIZE << NRM << std::endl;
        std::cout << YLW
                  << "\tEM3_RK45_DESIRED_TOL :" << em3::EM3_RK45_DESIRED_TOL
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_COMPD_MIN : ( :" << em3::EM3_COMPD_MIN[0]
                  << " ," << em3::EM3_COMPD_MIN[1] << ","
                  << em3::EM3_COMPD_MIN[2] << " )" << NRM << std::endl;
        std::cout << YLW << "\tEM3_COMPD_MAX : ( :" << em3::EM3_COMPD_MAX[0]
                  << " ," << em3::EM3_COMPD_MAX[1] << ","
                  << em3::EM3_COMPD_MAX[2] << " )" << NRM << std::endl;
        std::cout << YLW << "\tEM3_BLK_MIN : ( :" << em3::EM3_BLK_MIN_X << " ,"
                  << em3::EM3_BLK_MIN_Y << "," << em3::EM3_BLK_MIN_Z << " )"
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_BLK_MAX : ( :" << em3::EM3_BLK_MAX_X << " ,"
                  << em3::EM3_BLK_MAX_Y << "," << em3::EM3_BLK_MAX_Z << " )"
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_OCTREE_MIN : ( :" << em3::EM3_OCTREE_MIN[0]
                  << " ," << em3::EM3_OCTREE_MIN[1] << ","
                  << em3::EM3_OCTREE_MIN[2] << " )" << NRM << std::endl;
        std::cout << YLW << "\tEM3_OCTREE_MAX : ( :" << em3::EM3_OCTREE_MAX[0]
                  << " ," << em3::EM3_OCTREE_MAX[1] << ","
                  << em3::EM3_OCTREE_MAX[2] << " )" << NRM << std::endl;
        std::cout << YLW << "\tKO_DISS_SIGMA :" << em3::KO_DISS_SIGMA << NRM
                  << std::endl;
        std::cout << YLW << "\tEM3_ID_TYPE:" << em3::EM3_ID_TYPE << NRM
                  << std::endl;
        std::cout << YLW << "\tEM3_ID_AMP1:" << em3::EM3_ID_AMP1 << NRM
                  << std::endl;
        std::cout << YLW << "\tEM3_ID_LAMBDA1:" << em3::EM3_ID_LAMBDA1 << NRM
                  << std::endl;
        std::cout << YLW << "\tEM3_ID_AMP2:" << em3::EM3_ID_AMP2 << NRM
                  << std::endl;
        // std::cout<<YLW<<"\tEM3_ID_DELTA1:"<<em3::EM3_ID_DELTA1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_DELTA2:"<<em3::EM3_ID_DELTA2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_XC1:"<<em3::EM3_ID_XC1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_YC1:"<<em3::EM3_ID_YC1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_ZC1:"<<em3::EM3_ID_ZC1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_XC2:"<<em3::EM3_ID_XC2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_YC2:"<<em3::EM3_ID_YC2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_ZC2:"<<em3::EM3_ID_ZC2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_EPSX1:"<<em3::EM3_ID_EPSX1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_EPSY1:"<<em3::EM3_ID_EPSY1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_EPSZ1:"<<em3::EM3_ID_EPSY1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_EPSX2:"<<em3::EM3_ID_EPSX2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_EPSY2:"<<em3::EM3_ID_EPSY2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_EPSZ2:"<<em3::EM3_ID_EPSY2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_R1:"<<em3::EM3_ID_R1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_R2:"<<em3::EM3_ID_R2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_NU1:"<<em3::EM3_ID_NU1<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_NU2:"<<em3::EM3_ID_NU2<<NRM<<std::endl;
        // std::cout<<YLW<<"\tEM3_ID_OMEGA:"<<em3::EM3_ID_OMEGA<<NRM<<std::endl;

        // std::cout<<YLW<<"\tEM3_DIM :"<<em3::EM3_DIM<<NRM<<std::endl;
        std::cout << YLW << "\tEM3_MAXDEPTH :" << em3::EM3_MAXDEPTH << NRM
                  << std::endl;

        std::cout << YLW
                  << "\tEM3_NUM_REFINE_VARS :" << em3::EM3_NUM_REFINE_VARS
                  << NRM << std::endl;
        std::cout << YLW << "\tEM3_REFINE_VARIABLE_INDICES :[";
        for (unsigned int i = 0; i < em3::EM3_NUM_REFINE_VARS - 1; i++)
            std::cout << em3::EM3_REFINE_VARIABLE_INDICES[i] << ", ";
        std::cout
            << em3::EM3_REFINE_VARIABLE_INDICES[em3::EM3_NUM_REFINE_VARS - 1]
            << "]" << NRM << std::endl;

        std::cout << YLW
                  << "\tEM3_REFINEMENT_MODE :" << em3::EM3_REFINEMENT_MODE
                  << NRM << std::endl;

        std::cout << YLW << "\tEM3_NUM_EVOL_VARS_VTU_OUTPUT :"
                  << em3::EM3_NUM_EVOL_VARS_VTU_OUTPUT << NRM << std::endl;
        std::cout << YLW << "\tEM3_VTU_OUTPUT_EVOL_INDICES :[";
        for (unsigned int i = 0; i < em3::EM3_NUM_EVOL_VARS_VTU_OUTPUT - 1; i++)
            std::cout << em3::EM3_VTU_OUTPUT_EVOL_INDICES[i] << ", ";
        std::cout << em3::EM3_VTU_OUTPUT_EVOL_INDICES
                         [em3::EM3_NUM_EVOL_VARS_VTU_OUTPUT - 1]
                  << "]" << NRM << std::endl;

#ifdef EM3_USE_4TH_ORDER_DERIVS
        std::cout << "Using 4th order FD stencils. " << std::endl;
#endif

#ifdef EM3_USE_6TH_ORDER_DERIVS
        std::cout << "Using 6th order FD stencils. " << std::endl;
#endif

#ifdef EM3_USE_8TH_ORDER_DERIVS
        std::cout << "Using 8th order FD stencils. " << std::endl;
#endif
    }

    _InitializeHcurve(em3::EM3_DIM);
    m_uiMaxDepth = em3::EM3_MAXDEPTH;

    if (em3::EM3_NUM_VARS % em3::EM3_ASYNC_COMM_K != 0) {
        if (!rank)
            std::cout << "[overlap communication error]: total EM3_NUM_VARS: "
                      << em3::EM3_NUM_VARS
                      << " is not divisable by EM3_ASYNC_COMM_K: "
                      << em3::EM3_ASYNC_COMM_K << std::endl;
        exit(0);
    }

    // 2. generate the initial grid.
    std::vector<ot::TreeNode> tmpNodes;
    std::function<void(double, double, double, double*)> f_init =
        [](double x, double y, double z, double* var) {
            em3::initData(x, y, z, var);
        };

    const unsigned int interpVars = em3::EM3_NUM_VARS;
    unsigned int varIndex[interpVars];
    for (unsigned int i = 0; i < em3::EM3_NUM_VARS; i++) varIndex[i] = i;

    DendroIntL localSz, globalSz;
    double t_stat;
    double t_stat_g[3];

    em3::timer::t_f2o.start();

    if (em3::EM3_ENABLE_BLOCK_ADAPTIVITY) {
        if (!rank)
            std::cout << YLW << "Using block adaptive mesh. AMR disabled "
                      << NRM << std::endl;
        const Point pt_min(em3::EM3_BLK_MIN_X, em3::EM3_BLK_MIN_Y,
                           em3::EM3_BLK_MIN_Z);
        const Point pt_max(em3::EM3_BLK_MAX_X, em3::EM3_BLK_MAX_Y,
                           em3::EM3_BLK_MAX_Z);

        em3::blockAdaptiveOctree(tmpNodes, pt_min, pt_max, m_uiMaxDepth - 2,
                                 m_uiMaxDepth, comm);
    } else {
        if (!rank)
            std::cout << YLW << "Using function2Octree. AMR enabled " << NRM
                      << std::endl;
        function2Octree(f_init, em3::EM3_NUM_VARS,
                        em3::EM3_REFINE_VARIABLE_INDICES,
                        em3::EM3_NUM_REFINE_VARS, tmpNodes, m_uiMaxDepth - 2,
                        em3::EM3_WAVELET_TOL, em3::EM3_ELE_ORDER, comm);
    }
    em3::timer::t_f2o.stop();

    // some quick stats on how quickly that went
    t_stat = em3::timer::t_f2o.seconds;
    par::Mpi_Reduce(&t_stat, t_stat_g, 1, MPI_MIN, 0, comm);
    par::Mpi_Reduce(&t_stat, t_stat_g + 1, 1, MPI_SUM, 0, comm);
    par::Mpi_Reduce(&t_stat, t_stat_g + 2, 1, MPI_MAX, 0, comm);
    t_stat_g[1] = t_stat_g[1] / (double)npes;

    localSz = tmpNodes.size();
    par::Mpi_Reduce(&localSz, &globalSz, 1, MPI_SUM, 0, comm);

    if (!rank)
        std::cout << GRN << " function to octree max (s): " << t_stat_g[2]
                  << NRM << std::endl;
    if (!rank)
        std::cout << GRN << " function to octree # octants : " << globalSz
                  << NRM << std::endl;

    par::Mpi_Bcast(&globalSz, 1, 0, comm);
    const unsigned int grainSz =
        em3::EM3_DENDRO_GRAIN_SZ;  // DENDRO_DEFAULT_GRAIN_SZ;

    // FULL MESH GENERATION
    ot::Mesh* mesh =
        ot::createMesh(tmpNodes.data(), tmpNodes.size(), em3::EM3_ELE_ORDER,
                       comm, 1, ot::SM_TYPE::FDM, em3::EM3_DENDRO_GRAIN_SZ,
                       em3::EM3_LOAD_IMB_TOL, em3::EM3_SPLIT_FIX);
    mesh->setDomainBounds(
        Point(em3::EM3_GRID_MIN_X, em3::EM3_GRID_MIN_Y, em3::EM3_GRID_MIN_Z),
        Point(em3::EM3_GRID_MAX_X, em3::EM3_GRID_MAX_Y, em3::EM3_GRID_MAX_Z));
    unsigned int lmin, lmax;
    mesh->computeMinMaxLevel(lmin, lmax);
    if (!rank) {
        std::cout << "================= Grid Info (Before init grid "
                     "converge):==============================================="
                     "========"
                  << std::endl;
        std::cout << "lmin: " << lmin << " lmax:" << lmax << std::endl;
        std::cout << "dx: "
                  << ((em3::EM3_COMPD_MAX[0] - em3::EM3_COMPD_MIN[0]) *
                      ((1u << (m_uiMaxDepth - lmax)) /
                       ((double)em3::EM3_ELE_ORDER)) /
                      ((double)(1u << (m_uiMaxDepth))))
                  << std::endl;
        std::cout << "dt: "
                  << em3::EM3_CFL_FACTOR *
                         ((em3::EM3_COMPD_MAX[0] - em3::EM3_COMPD_MIN[0]) *
                          ((1u << (m_uiMaxDepth - lmax)) /
                           ((double)em3::EM3_ELE_ORDER)) /
                          ((double)(1u << (m_uiMaxDepth))))
                  << std::endl;
        std::cout << "ts mode: " << ts_mode << std::endl;
        std::cout << "========================================================="
                     "======================================================"
                  << std::endl;
    }
    em3::EM3_RK45_TIME_STEP_SIZE =
        em3::EM3_CFL_FACTOR *
        ((em3::EM3_COMPD_MAX[0] - em3::EM3_COMPD_MIN[0]) *
         ((1u << (m_uiMaxDepth - lmax)) / ((double)em3::EM3_ELE_ORDER)) /
         ((double)(1u << (m_uiMaxDepth))));
    tmpNodes.clear();

    if (ts_mode == 1) {
        em3::EM3Ctx* em3Ctx = new em3::EM3Ctx(mesh);

        ts::ETS<DendroScalar, em3::EM3Ctx>* ets =
            new ts::ETS<DendroScalar, em3::EM3Ctx>(em3Ctx);
        ets->set_evolve_vars(em3Ctx->get_evolution_vars());

        if ((RKType)em3::EM3_RK_TYPE == RKType::RK3)
            ets->set_ets_coefficients(ts::ETSType::RK3);
        else if ((RKType)em3::EM3_RK_TYPE == RKType::RK4)
            ets->set_ets_coefficients(ts::ETSType::RK4);
        else if ((RKType)em3::EM3_RK_TYPE == RKType::RK45)
            ets->set_ets_coefficients(ts::ETSType::RK5);

        ets->init();

#if defined __PROFILE_CTX__ && defined __PROFILE_ETS__
        std::ofstream outfile;
        char fname[256];
        sprintf(fname, "em3Ctx_%d.txt", npes);
        if (!rank) {
            outfile.open(fname, std::ios_base::app);
            time_t now = time(0);
            // convert now to string form
            char* dt = ctime(&now);
            outfile << "======================================================="
                       "====="
                    << std::endl;
            outfile << "Current time : " << dt << " --- " << std::endl;
            outfile << "======================================================="
                       "====="
                    << std::endl;
        }

        ets->init_pt();
        em3Ctx->reset_pt();
        ets->dump_pt(outfile);
#endif

        double t1 = MPI_Wtime();

        while (ets->curr_time() < em3::EM3_RK45_TIME_END) {
            const DendroIntL step = ets->curr_step();

            const DendroScalar time = ets->curr_time();

            std::cout << step << " " << time << std::endl;

            // global variables that can be used elsewhere...
            // em3::EM3_CURRENT_RK_COORD_TIME = time;
            // em3::EM3_CURRENT_RK_STEP = step;

            const bool isActive = ets->is_active();
            const unsigned int rank_global = ets->get_global_rank();

            if ((step % em3::EM3_REMESH_TEST_FREQ) == 0) {
                bool isRemesh = em3Ctx->is_remesh();
                if (isRemesh) {
                    if (!rank_global)
                        std::cout << GRN << "[ETS] : Remesh was triggered!"
                                  << NRM << std::endl;

                    em3Ctx->remesh_and_gridtransfer(em3::EM3_DENDRO_GRAIN_SZ,
                                                    em3::EM3_LOAD_IMB_TOL,
                                                    em3::EM3_SPLIT_FIX);
                    em3::deallocate_em3_deriv_workspace();
                    em3::allocate_em3_deriv_workspace(em3Ctx->get_mesh(), 1);
                    ets->sync_with_mesh();

                    ot::Mesh* pmesh = em3Ctx->get_mesh();
                    unsigned int lmin, lmax;
                    pmesh->computeMinMaxLevel(lmin, lmax);
                    if (!pmesh->getMPIRank())
                        printf("  ... New Grid Levels (min, max) = (%d, %d)\n",
                               lmin, lmax);
                    em3::EM3_RK45_TIME_STEP_SIZE =
                        em3::EM3_CFL_FACTOR *
                        ((em3::EM3_COMPD_MAX[0] - em3::EM3_COMPD_MIN[0]) *
                         ((1u << (m_uiMaxDepth - lmax)) /
                          ((double)em3::EM3_ELE_ORDER)) /
                         ((double)(1u << (m_uiMaxDepth))));
                    ts::TSInfo ts_in = em3Ctx->get_ts_info();
                    ts_in._m_uiTh = em3::EM3_RK45_TIME_STEP_SIZE;
                    em3Ctx->set_ts_info(ts_in);
                }
            }

            if ((step % em3::EM3_IO_OUTPUT_FREQ) == 0) {
                std::cout << "Attempting?" << std::endl;
                if (!rank_global)
                    std::cout
                        << "[ETS] : Executing step :  " << ets->curr_step()
                        << "\tcurrent time :" << ets->curr_time()
                        << "\t dt:" << ets->ts_size() << "\t" << std::endl;

                em3Ctx->write_vtu();
                // termainal output should be called after write_vtu because
                // that's what triggers all of the additional stuff
                em3Ctx->terminal_output();
            }

            if ((step % em3::EM3_CHECKPT_FREQ) == 0) {
                em3Ctx->write_checkpt();
            }

            ets->evolve();
        }

#if defined __PROFILE_CTX__ && defined __PROFILE_ETS__
        ets->dump_pt(outfile);
        // em3Ctx->dump_pt(outfile);
#endif

        double t2 = MPI_Wtime() - t1;
        double t2_g;
        par::Mpi_Allreduce(&t2, &t2_g, 1, MPI_MAX, ets->get_global_comm());
        if (!(ets->get_global_rank())) {
            std::cout << " ETS time (max) : " << t2_g << std::endl;
            std::cout << std::endl
                      << std::endl
                      << GRN << "==============================" << std::endl;
            std::cout << GRN << "======= SOLVER FINISHED ======" << std::endl;
            std::cout << "==============================" << NRM << std::endl;
        }

        delete em3Ctx->get_mesh();
        delete em3Ctx;
        delete ets;
    }

    MPI_Finalize();
    return 0;
}