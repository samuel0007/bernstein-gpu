// Linear solver for the 2D planewave problem
// - structured mesh
// - first-order Sommerfeld ABC
// ==========================================
// Copyright (C) 2022 Adeeb Arif Kor

#include "Linear.hpp"
#include "planar_wave_triangles_gpu.h"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <iomanip>
#include <iostream>

#define T_MPI MPI_DOUBLE
using T = double;

int main(int argc, char* argv[]) {
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Source parameters
    const T sourceFrequency = 0.5e6;      // (Hz)
    const T sourceAmplitude = 60000;      // (Pa)
    const T period = 1 / sourceFrequency; // (s)

    // Material parameters
    const T speedOfSound = 1500; // (m/s)
    const T density = 1000;      // (kg/m^3)

    // Domain parameters
    const T domainLength = 0.12; // (m)

    // FE parameters
    const int degreeOfBasis = 2;

    // Read mesh and mesh tags
    auto coord_element = fem::CoordinateElement<T>(mesh::CellType::triangle, 1);
    io::XDMFFile fmesh(MPI_COMM_WORLD, std::string(DATA_DIR)+"/mesh.xdmf", "r");
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        fmesh.read_mesh(coord_element, mesh::GhostMode::none, "planewave_2d_1_t"));
    mesh->topology()->create_connectivity(1, 2);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "planewave_2d_1_t_cells"));
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "planewave_2d_1_t_facets"));

    // Mesh parameters
    const int tdim = mesh->topology()->dim();
    const int num_cell = mesh->topology()->index_map(tdim)->size_local();
    std::vector<int> num_cell_range(num_cell);
    std::iota(num_cell_range.begin(), num_cell_range.end(), 0.0);
    std::vector<T> mesh_size_local = mesh::h(*mesh, num_cell_range, tdim);
    std::vector<T>::iterator min_mesh_size_local
        = std::min_element(mesh_size_local.begin(), mesh_size_local.end());
    int mesh_size_local_idx = std::distance(mesh_size_local.begin(), min_mesh_size_local);
    T meshSizeMinLocal = mesh_size_local.at(mesh_size_local_idx);
    T meshSizeMinGlobal;
    MPI_Reduce(&meshSizeMinLocal, &meshSizeMinGlobal, 1, T_MPI, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Bcast(&meshSizeMinGlobal, 1, T_MPI, 0, MPI_COMM_WORLD);

    // Finite element
    basix::FiniteElement element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::triangle, degreeOfBasis,
      basix::element::lagrange_variant::bernstein,
      basix::element::dpc_variant::unset, false
    );

    // Define DG function space for the physical parameters of the domain
    basix::FiniteElement element_DG = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::triangle, 0,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, true
    );
    auto V_DG = std::make_shared<fem::FunctionSpace<T>>(
        fem::create_functionspace(mesh, element_DG));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);

    auto cells_1 = mt_cell->find(1);

    std::span<T> c0_ = c0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { c0_[i] = speedOfSound; });
    c0->x()->scatter_fwd();

    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t& i) { rho0_[i] = density; });
    rho0->x()->scatter_fwd();

    // Temporal parameters
    const T CFL = 0.9;
    T timeStepSize = CFL * meshSizeMinGlobal / (speedOfSound * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const T startTime = 0.0;
    // const T finalTime = domainLength / speedOfSound + 4.0 / sourceFrequency;
    const T finalTime = (domainLength / speedOfSound + 4.0 / sourceFrequency) / 4.;
    const int numberOfStep = (finalTime - startTime) / timeStepSize + 1;

    if (mpi_rank == 0) {
      std::cout << "Problem type: Planewave 2D"
                << "\n";
      std::cout << "Speed of sound: " << speedOfSound << "\n";
      std::cout << "Density: " << density << "\n";
      std::cout << "Source frequency: " << sourceFrequency << "\n";
      std::cout << "Source amplitude: " << sourceAmplitude << "\n";
      std::cout << "Domain length: " << domainLength << "\n";
      std::cout << "Polynomial basis degree: " << degreeOfBasis << "\n";
      std::cout << "Minimum mesh size: ";
      std::cout << std::setprecision(2) << meshSizeMinGlobal << "\n";
      std::cout << "CFL number: " << CFL << "\n";
      std::cout << "Time step size: " << timeStepSize << "\n";
      std::cout << "Number of steps per period: " << stepPerPeriod << "\n";
      std::cout << "Total number of steps: " << numberOfStep << "\n";
    }

    // Model
    auto model = LinearSpectral<T, 4>(element, mesh, mt_facet, c0, rho0, sourceFrequency,
                                      sourceAmplitude, speedOfSound);

    // Solve
    common::Timer tsolve("Solve time");

    model.init();

    tsolve.start();
    model.rk4(startTime, finalTime, timeStepSize);
    tsolve.stop();

    if (mpi_rank == 0) {
      std::cout << "Solve time: " << tsolve.elapsed()[0] << "s" << std::endl;
      std::cout << "Time per step: " << tsolve.elapsed()[0] / numberOfStep << "s" << std::endl;
    }

    // Final solution
    auto u_n = model.u_sol();

    // Output to VTX
    dolfinx::io::VTXWriter<T> u_out(mesh->comm(), "output_final.bp", {u_n}, "bp5");
    u_out.write(0.0);

    // Check norms
    auto Norm = std::make_shared<fem::Form<T>>(
        fem::create_form<T, T>(*form_planar_wave_triangles_gpu_Norm, {}, {{"u_n", u_n}}, {}, {}, {}, mesh));
    T norm = fem::assemble_scalar(*Norm);

    if (mpi_rank == 0) {
      std::cout << "L2 norm: " << std::sqrt(norm) << "\n";
    }
  }
}