#pragma once
#include <basix/finite-element.h>
#include <boost/program_options.hpp>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <unordered_map>

#include "types.hpp"

namespace po = boost::program_options;
namespace el = basix::element;

constexpr const char *logo = R"(   ___                ___          
  / __\ __ ___  ___  / __\   _ ___ 
 / _\| '__/ _ \/ _ \/ _\| | | / __|
/ /  | | |  __/  __/ /  | |_| \__ \
\/   |_|  \___|\___\/    \__,_|___/
                                   )";

static const std::unordered_map<std::string, el::lagrange_variant>
    str_to_basixlv = {{"bernstein", el::lagrange_variant::bernstein},
                      {"gll_warped", el::lagrange_variant::gll_warped}};

template <typename T> void display_user_config(const UserConfig<T> &cfg) {
  spdlog::info("Mesh name              : {}", cfg.mesh_name);
  spdlog::info("Mesh filepath          : {}", cfg.mesh_filepath);
  spdlog::info("Output file            : {}", cfg.output_filepath);
  spdlog::info("Material Case          : {}",
               static_cast<int>(cfg.material_case));
  spdlog::info("Model type             : {}", static_cast<int>(cfg.model_type));
  spdlog::info("CFL number             : {}", cfg.CFL);
  spdlog::info("Source freq (Hz)       : {}", cfg.source_frequency);
  spdlog::info("Source amp (Pa)        : {}", cfg.source_amplitude);
  spdlog::info("Domain length (m)      : {}", cfg.domain_length);
  spdlog::info("Output steps           : {}", cfg.output_steps);
  spdlog::info("In-situ output enabled : {}", cfg.insitu);
  spdlog::info("In-situ output steps   : {}", cfg.insitu_output_steps);
  spdlog::info("CG tolerance           : {}", cfg.cg_tol);
  spdlog::info("CG max iterations      : {}", cfg.cg_max_steps);
  spdlog::info("Window length (periods): {}", cfg.window_length);
  spdlog::info("Log level              : {}",
               spdlog::level::to_string_view(cfg.log_level));
}

template <typename U>
UserConfig<U> make_user_config(const po::variables_map &vm) {
  namespace fs = std::filesystem;
  UserConfig<U> cfg;
  cfg.mesh_name = vm["mesh"].as<std::string>();
  cfg.mesh_filepath = fs::path(DATA_DIR) / cfg.mesh_name / "mesh.xdmf";
  cfg.mesh_dir = fs::path(DATA_DIR) / cfg.mesh_name;
  cfg.lvariant = str_to_basixlv.at(vm["polynomial-basis"].as<std::string>());
  cfg.output_filepath = fs::path(DATA_DIR) / cfg.mesh_name / vm["output-path"].as<std::string>();
  {
    auto ext = fs::path(cfg.output_filepath).extension();
    if (ext != ".bp")
      throw std::invalid_argument("output_path must end with .bp");
  }

  cfg.CFL = vm["CFL"].as<U>();
  cfg.source_frequency = vm["source-frequency"].as<U>();
  cfg.source_amplitude = vm["source-amplitude"].as<U>();
  cfg.domain_length = vm["domain-length"].as<U>();
  cfg.domain_width = vm["domain-width"].as<U>();
  cfg.output_steps = vm["output-steps"].as<int>();
  cfg.log_level = spdlog::level::from_str(vm["log-level"].as<std::string>());
  cfg.material_case = static_cast<MaterialCase>(vm["material-case"].as<int>());
  cfg.sample_harmonic = vm["sample-harmonic"].as<int>();
  cfg.sampling_periods = vm["sampling-periods"].as<int>();

  cfg.insitu = vm["insitu"].as<bool>();
  cfg.insitu_output_steps = vm["insitu-output-steps"].as<int>();
  cfg.insitu_with_yaml = vm["insitu-with-yaml"].as<bool>();
  cfg.model_type = static_cast<ModelType>(vm["model-type"].as<int>());
  cfg.timestepping_type =
      static_cast<TimesteppingType>(vm["timestepping-type"].as<int>());
  if (static_cast<int>(cfg.model_type) !=
      static_cast<int>(cfg.timestepping_type)) {
    // TODO: maybe only expose single "model-timestepper" combo, as whatsoever
    // the classes are coupled at the mathematical level. Maybe remove model
    // type concept and only have an "operator provider" concept.
    throw std::runtime_error("Invalid model and timestepper configuration");
  }
  cfg.cg_tol = vm["cg-tol"].as<double>();
  cfg.cg_max_steps = vm["cg-maxsteps"].as<int>();
  cfg.nonlinear_tol = vm["nonlinear-tol"].as<double>();
  cfg.window_length = vm["window-length"].as<double>();

  cfg.sample_nx = vm["sample-nx"].as<int>();
  cfg.sample_nz = vm["sample-nz"].as<int>();

  return cfg;
}

template <typename T>
po::variables_map parse_cli_config(int argc, char *argv[]) {

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()("help,h", "print usage message")
      ("mesh,m", po::value<std::string>()->default_value("BP1-small"), "mesh folder name")
      ("output-path,o", po::value<std::string>()->default_value("output.bp"), "output path, must end in .bp")
      ("polynomial-basis", po::value<std::string>()->default_value("gll_warped"), "Polynomial basis: bernstein, gll_warped")
      ("material-case", po::value<int>()->default_value(1), "Material case [1-7]")
      ("model-type", po::value<int>()->default_value(1), "Model type [1-2]")
      ("timestepping-type", po::value<int>()->default_value(1), "Timestepping type [1-2]")
      ("sample-harmonic", po::value<int>()->default_value(1), "Number of harmonics to sample at the end of the simulation.")
      ("CFL", po::value<T>()->default_value(0.5), "CFL number")
      ("source-frequency", po::value<T>()->default_value(0.1e6), "Source frequency (Hz)")
      ("source-amplitude", po::value<T>()->default_value(60000), "Source amplitude (Pa)")
      ("domain-length", po::value<T>()->default_value(0.12), "Domain length (m)")
      ("domain-width", po::value<T>()->default_value(0.07), "Domain width (m)")
      ("window-length", po::value<T>()->default_value(4), "Window length (periods)")
      ("output-steps", po::value<int>()->default_value(200), "Frequency of I/O output steps")
      ("insitu", po::value<bool>()->default_value(true), "Insitu visualisation")
      ("insitu-output-steps", po::value<int>()->default_value(50), "Number of insitu output steps")
      ("insitu-with-yaml", po::value<bool>()->default_value(true), "Search for an ascent_actions.yaml file in mesh dir.")
      ("cg-tol", po::value<T>()->default_value(1e-8), "Tolerance of CG solver")
      ("cg-maxsteps", po::value<int>()->default_value(200), "Max number of CG iterations")
      ("nonlinear-tol", po::value<T>()->default_value(1e-6), "Tolerance of nonlinear solver")
      ("sample-nx", po::value<int>()->default_value(200), "Sampling in X direction")
      ("sample-nz", po::value<int>()->default_value(200), "Sampling in Z direction")
      ("sampling-periods", po::value<int>()->default_value(4), "Sampling time (periods)")
      ("log-level", po::value<std::string>()->default_value("info"),
       "Log level: trace, debug, info, warn, err, critical, off");
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

std::string asciiLogo() {
  return std::string("\n\033[38;2;142;232;216m") + logo + "\033[0m\n" +
         "Focused Ultrasounds High Order Matrix Free FEM GPU solver for "
         "unstructured meshes\n";
}
