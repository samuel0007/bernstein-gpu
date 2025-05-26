// Linear solver for the 2D planewave problem
// - structured mesh
// - first-order Sommerfeld ABC
// ==========================================
// Copyright (C) 2022 Adeeb Arif Kor

#include "Linear.hpp"
#include "spherical_transducer_gpu.h"

#include <cmath>
#include <dolfinx.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/io/XDMFFile.h>
#include <iomanip>
#include <iostream>

#include <boost/program_options.hpp>

#define T_MPI MPI_DOUBLE // todo
using T = SCALAR_TYPE;

#if USE_HIP
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;
#elif USE_CUDA
using DeviceVector = dolfinx::acc::Vector<T, acc::Device::CUDA>;
#else
static_assert(false)
#endif

using namespace dolfinx;
namespace po = boost::program_options;

po::variables_map get_cli_config(int argc, char *argv[]) {


  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()("help,h", "print usage message")
      ("mesh,m", po::value<std::string>()->default_value("mesh"), "mesh file name")
      ("output,o", po::value<std::string>()->default_value("output"), "output file name")
      ("CFL", po::value<T>()->default_value(0.5), "CFL number")
      ("source-frequency", po::value<T>()->default_value(0.5e6), "Source frequency (Hz)")
      ("source-amplitude", po::value<T>()->default_value(60000), "Source amplitude (Pa)")
      ("speed-of-sound", po::value<T>()->default_value(1500), "Speed of sound (m/s)")
      ("density", po::value<T>()->default_value(1000), "Density (kg/m^3)")
      ("output-steps", po::value<int>()->default_value(50), "Number of output steps");
  // clang-format on

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  return vm;
}

int main(int argc, char *argv[]) {
  auto vm = get_cli_config(argc, argv);

  if (vm.count("help")) {
    std::cout << "Use --mesh <filename>\n";
    return 0;
  }
  const std::string mesh_filepath = std::filesystem::path(DATA_DIR) /
                                    (vm["mesh"].as<std::string>() + ".xdmf");
  const std::string output_filepath = std::filesystem::path(DATA_DIR) /
                                    (vm["output"].as<std::string>() + ".bp");
  const T CFL = vm["CFL"].as<T>();
  const T sourceFrequency = vm["source-frequency"].as<T>();
  const T sourceAmplitude = vm["source-amplitude"].as<T>();
  const T speedOfSound = vm["speed-of-sound"].as<T>(); // (m/s)
  const T density = vm["density"].as<T>();             // (kg/m^3)
  const int outputSteps = vm["output-steps"].as<int>();

  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);
  {
    // MPI
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Source parameters
    const T period = 1 / sourceFrequency; // (s)

    // Domain parameters
    const T domainLength = 0.12; // (m)

    // FE parameters
    const int degreeOfBasis = 2;

    // Read mesh and mesh tags
    auto coord_element =
        fem::CoordinateElement<T>(mesh::CellType::tetrahedron, 1);

    io::XDMFFile fmesh(MPI_COMM_WORLD, mesh_filepath, "r");
    auto mesh = std::make_shared<mesh::Mesh<T>>(
        fmesh.read_mesh(coord_element, mesh::GhostMode::none, "mesh"));
    mesh->topology()->create_connectivity(2, 3);
    auto mt_cell = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "Cell tags", std::nullopt));
    auto mt_facet = std::make_shared<mesh::MeshTags<std::int32_t>>(
        fmesh.read_meshtags(*mesh, "Facet tags", std::nullopt));

    // Mesh parameters
    const int tdim = mesh->topology()->dim();
    const int num_cell = mesh->topology()->index_map(tdim)->size_local();
    std::vector<int> num_cell_range(num_cell);
    std::iota(num_cell_range.begin(), num_cell_range.end(), 0.0);
    std::vector<T> mesh_size_local = mesh::h(*mesh, num_cell_range, tdim);
    std::vector<T>::iterator min_mesh_size_local =
        std::min_element(mesh_size_local.begin(), mesh_size_local.end());
    int mesh_size_local_idx =
        std::distance(mesh_size_local.begin(), min_mesh_size_local);
    T meshSizeMinLocal = mesh_size_local.at(mesh_size_local_idx);
    T meshSizeMinGlobal;
    MPI_Reduce(&meshSizeMinLocal, &meshSizeMinGlobal, 1, T_MPI, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Bcast(&meshSizeMinGlobal, 1, T_MPI, 0, MPI_COMM_WORLD);

    // Finite element
    basix::FiniteElement element = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::tetrahedron,
        degreeOfBasis, basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    // Define DG function space for the physical parameters of the domain
    basix::FiniteElement element_DG = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::tetrahedron, 0,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, true);

    auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
        mesh, std::make_shared<const fem::FiniteElement<T>>(element)));
    auto V_DG =
        std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
            mesh, std::make_shared<const fem::FiniteElement<T>>(element_DG)));
    auto c0 = std::make_shared<fem::Function<T>>(V_DG);
    auto rho0 = std::make_shared<fem::Function<T>>(V_DG);
    auto alpha = std::make_shared<fem::Function<T>>(V_DG);

    auto cells_1 = mt_cell->find(1);

    std::cout << mt_facet->values().size() << std::endl;

    std::span<T> c0_ = c0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t &i) { c0_[i] = speedOfSound; });
    c0->x()->scatter_fwd();

    std::span<T> rho0_ = rho0->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(),
                  [&](std::int32_t &i) { rho0_[i] = density; });
    rho0->x()->scatter_fwd();

    std::span<T> alpha_ = alpha->x()->mutable_array();
    std::for_each(cells_1.begin(), cells_1.end(), [&](std::int32_t &i) {
      alpha_[i] = 1. / (rho0_[i] * c0_[i] * c0_[i]);
    });
    alpha->x()->scatter_fwd();
    std::cout << "cells_1 size:" << cells_1.size() << std::endl;
    std::cout << "alpha size:" << alpha->x()->mutable_array().size()
              << std::endl;

    // Temporal parameters
    T timeStepSize = CFL * meshSizeMinGlobal /
                     (speedOfSound * degreeOfBasis * degreeOfBasis);
    const int stepPerPeriod = period / timeStepSize + 1;
    timeStepSize = period / stepPerPeriod;
    const T startTime = 0.0;
    const T finalTime = (domainLength / speedOfSound + 4.0 / sourceFrequency);
    const int numberOfStep = (finalTime - startTime) / timeStepSize + 1;

    if (mpi_rank == 0) {
      std::cout << "Problem type: Spherical Transducer 3D" << "\n";
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
      std::cout << "Number of cells" << num_cell << "\n";
    }

    // Model
    auto model = LinearSpectral<T, degreeOfBasis, DeviceVector>(
        mesh, V, mt_facet, c0, rho0, alpha, sourceFrequency, sourceAmplitude,
        speedOfSound);

    // Output space
    basix::FiniteElement lagrange_element = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::tetrahedron,
        degreeOfBasis, basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    auto V_out =
        std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
            mesh,
            std::make_shared<const fem::FiniteElement<T>>(lagrange_element)));

    auto u_out = std::make_shared<fem::Function<T>>(V_out);

    // Output to VTX
    io::VTXWriter<T> f_out(mesh->comm(), output_filepath, {u_out}, "bp5");

    model.init();

    // tsolve.start();
    model.rk4(startTime, finalTime, timeStepSize, outputSteps, u_out, f_out);
  }
}