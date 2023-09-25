project(EM3)

# only EM3 options
option(EM3_COMPUTE_CONSTRAINTS "Enable constraint equation computation" ON)
option(EM3_USE_4TH_ORDER_DERIVS "4th order derivs" OFF)
option(EM3_USE_6TH_ORDER_DERIVS "6th order derivs" ON)
option(EM3_USE_8TH_ORDER_DERIVS "8th order derivs" OFF)

if(EM3_USE_4TH_ORDER_DERIVS)
    set(EM3_USE_6TH_ORDER_DERIVS OFF)
    set(EM3_USE_8TH_ORDER_DERIVS OFF)
elseif(EM3_USE_6TH_ORDER_DERIVS)
    set(EM3_USE_4TH_ORDER_DERIVS OFF)
    set(EM3_USE_8TH_ORDER_DERIVS OFF)
elseif(EM3_USE_8TH_ORDER_DERIVS)
    set(EM3_USE_4TH_ORDER_DERIVS OFF)
    set(EM3_USE_6TH_ORDER_DERIVS OFF)
endif()

if(EM3_USE_4TH_ORDER_DERIVS)
    add_definitions(-DEM3_USE_4TH_ORDER_DERIVS)
endif()

if(EM3_USE_6TH_ORDER_DERIVS)
    add_definitions(-DEM3_USE_6TH_ORDER_DERIVS)
endif()

if(EM3_USE_8TH_ORDER_DERIVS)
    add_definitions(-DEM3_USE_8TH_ORDER_DERIVS)
endif()

#add_definitions(-DDEBUG_RK_SOLVER) 

if(EM3_COMPUTE_CONSTRAINTS)
    add_definitions(-DEM3_COMPUTE_CONSTRAINTS)
endif() 

set(EM3_INC include/parameters.h
        include/rk4em3.h
        include/em3Utils.h
        include/rhs.h
        include/kim.h
        include/compact_derivs.h
        include/derivs.h
        include/profile_params.h
        include/physcon.h 
        include/em3Ctx.h
        include/JTP6.h
        include/HAMR.h
        include/kim_filter.h
        )

        set(EM3_SRC src/rk4em3.cpp
        src/parameters.cpp
        src/em3Utils.cpp
        src/rhs.cpp
        src/compact_derivs.cpp
        src/derivs.cpp
        src/kim.cpp
        src/HAMR.cpp
        src/profile_params.cpp
        src/physcon.cpp 
        src/JTP6.cpp
        src/kim_filter.cpp
        #src/em3Ctx.cpp
        )

set(SOURCE_FILES src/em3.cpp  include/em3.h ${EM3_INC} ${EM3_SRC})
add_executable(em3Solver ${SOURCE_FILES})
target_include_directories(em3Solver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(em3Solver PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/include)
target_include_directories(em3Solver PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/include/test)
target_include_directories(em3Solver PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/examples/include)
target_include_directories(em3Solver PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/FEM/include)
target_include_directories(em3Solver PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/ODE/include)
target_include_directories(em3Solver PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/LinAlg/include)
target_include_directories(em3Solver PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/IO/vtk/include)
target_include_directories(em3Solver PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/IO/zlib/inc)
target_include_directories(em3Solver PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(em3Solver PRIVATE ${GSL_INCLUDE_DIRS})
if(WITH_CUDA)
    target_include_directories(em3Solver PRIVATE ${CUDA_INCLUDE_DIRS})
endif()
target_link_libraries(em3Solver dendro5 ${LAPACK_LIBRARIES} ${MPI_LIBRARIES} m)
target_link_libraries(em3Solver xsmm)


# =========== CTX VERSION OF SOLVEr

set(SOURCE_FILES include/em3Ctx.h src/em3Ctx.cpp src/em3_main.cpp  include/em3.h ${EM3_INC} ${EM3_SRC})
add_executable(em3SolverNEW ${SOURCE_FILES})
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/include)
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/include/test)
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/examples/include)
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/FEM/include)
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/ODE/include)
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/LinAlg/include)
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/IO/vtk/include)
target_include_directories(em3SolverNEW PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/IO/zlib/inc)
target_include_directories(em3SolverNEW PRIVATE ${MPI_INCLUDE_PATH})
target_include_directories(em3SolverNEW PRIVATE ${GSL_INCLUDE_DIRS})
if(WITH_CUDA)
    target_include_directories(em3SolverNEW PRIVATE ${CUDA_INCLUDE_DIRS})
endif()
target_link_libraries(em3SolverNEW dendro5 ${LAPACK_LIBRARIES} ${MPI_LIBRARIES} m)
target_link_libraries(em3SolverNEW xsmm)

# END CTX VERSION OF SOLVER



set(DERIV_TEST_SRC src/compact_derivs.cpp
        src/derivs.cpp
        src/kim.cpp
        src/HAMR.cpp
        src/JTP6.cpp
        src/kim_filter.cpp)

set(DERIV_TEST_INC include/rhs.h
        include/kim.h
        include/compact_derivs.h
        include/JTP6.h
        include/HAMR.h
        include/kim_filter.h)

# message(${BLAS_LIBRARIES})
# message(${BLAS_LINKER_FLAGS})
# message(${TARGET_FILE:xsmm})

# add_definitions(-DBLAS=0)


set(SOURCE_FILES src/test_cfd_profiling.cpp ${DERIV_TEST_INC} ${DERIV_TEST_SRC})

add_executable(derivTesting ${SOURCE_FILES})
target_include_directories(derivTesting PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(derivTesting PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/include)
target_include_directories(derivTesting PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(derivTesting xsmm) # "-Wl,--wrap=dgemm_,--wrap=sgemm_ ${BLAS_LIBRARIES} $<TARGET_FILE:xsmm>")

target_link_libraries(derivTesting dendro5 ${LAPACK_LIBRARIES} ${MPI_LIBRARIES} m)
# message("-Wl,--wrap=dgemm_,--wrap=sgemm_" ${BLAS_INCLUDE_DIR} $<TARGET_FILE:xsmm>)


# set(SOURCE_FILES src/em3_lts.cpp  include/em3.h ${EM3_INC} ${EM3_SRC})
# add_executable(em3SolverNUTS ${SOURCE_FILES})
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/include)
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/include/test)
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/examples/include)
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/FEM/include)
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/ODE/include)
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/LinAlg/include)
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/IO/vtk/include)
# target_include_directories(em3SolverNUTS PRIVATE ${CMAKE_SOURCE_DIR}/dendrolib/IO/zlib/inc)
# target_include_directories(em3SolverNUTS PRIVATE ${MPI_INCLUDE_PATH})
# target_include_directories(em3SolverNUTS PRIVATE ${GSL_INCLUDE_DIRS})
# if(WITH_CUDA)
#     target_include_directories(em3SolverNUTS PRIVATE ${CUDA_INCLUDE_DIRS})
# endif()
# target_link_libraries(em3SolverNUTS dendro5 ${LAPACK_LIBRARIES} ${MPI_LIBRARIES} m)
