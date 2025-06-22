#pragma once
#include <spdlog/spdlog.h>
#include <boost/program_options.hpp>
#include <filesystem>

namespace po = boost::program_options;

constexpr const char *logo = R"(   ___                ___          
  / __\ __ ___  ___  / __\   _ ___ 
 / _\| '__/ _ \/ _ \/ _\| | | / __|
/ /  | | |  __/  __/ /  | |_| \__ \
\/   |_|  \___|\___\/    \__,_|___/
                                   )";

template <typename T>
struct UserConfig
{
  std::string mesh_filepath;
  std::string output_filepath;
  T CFL;
  T sourceFrequency;
  T sourceAmplitude;
  T speedOfSound;
  T density;
  T domainLength;
  int outputSteps;
  spdlog::level::level_enum log_level;
};


template<typename T>
void display_user_config(const UserConfig<T>& cfg)
{
    spdlog::info("Mesh file        : {}", cfg.mesh_filepath);
    spdlog::info("Output file      : {}", cfg.output_filepath);
    spdlog::info("CFL number       : {}", cfg.CFL);
    spdlog::info("Source freq (Hz) : {}", cfg.sourceFrequency);
    spdlog::info("Source amp (Pa)  : {}", cfg.sourceAmplitude);
    spdlog::info("Speed of sound   : {}", cfg.speedOfSound);
    spdlog::info("Density (kg/m³)  : {}", cfg.density);
    spdlog::info("Domain length (m): {}", cfg.domainLength);
    spdlog::info("Output steps     : {}", cfg.outputSteps);
    spdlog::info("Log level        : {}", spdlog::level::to_string_view(cfg.log_level));
}

template <typename T>
UserConfig<T> make_user_config(const po::variables_map &vm)
{
  namespace fs = std::filesystem;
  UserConfig<T> cfg;
  cfg.mesh_filepath = vm["mesh_path"].as<std::string>();
  {
    auto ext = fs::path(cfg.mesh_filepath).extension();
    if (ext != ".xdmf")
      throw std::invalid_argument("mesh_path must end with .xdmf");
  }

  cfg.output_filepath = vm["output_path"].as<std::string>();
  {
    auto ext = fs::path(cfg.output_filepath).extension();
    if (ext != ".bp")
      throw std::invalid_argument("output_path must end with .bp");
  }

  cfg.CFL = vm["CFL"].as<T>();
  cfg.sourceFrequency = vm["source-frequency"].as<T>();
  cfg.sourceAmplitude = vm["source-amplitude"].as<T>();
  cfg.speedOfSound = vm["speed-of-sound"].as<T>();
  cfg.density = vm["density"].as<T>();
  cfg.domainLength = vm["domain-length"].as<T>();
  cfg.outputSteps = vm["output-steps"].as<int>();

  auto lvl = vm["log-level"].as<std::string>();
  cfg.log_level = spdlog::level::from_str(lvl);

  return cfg;
}

template <typename T>
po::variables_map get_cli_config(int argc, char *argv[])
{

  po::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()("help,h", "print usage message")
      ("mesh_path,m", po::value<std::string>()->default_value("mesh.xdmf"), "mesh path, must end in .xdmf")
      ("output_path,o", po::value<std::string>()->default_value("output.bp"), "output path, must end in .bp")
      ("CFL", po::value<T>()->default_value(0.5), "CFL number")
      ("source-frequency", po::value<T>()->default_value(0.5e6), "Source frequency (Hz)")
      ("source-amplitude", po::value<T>()->default_value(60000), "Source amplitude (Pa)")
      ("speed-of-sound", po::value<T>()->default_value(1500), "Speed of sound (m/s)")
      ("density", po::value<T>()->default_value(1000), "Density (kg/m^3)")
      ("domain-length", po::value<T>()->default_value(0.12), "Domain length (m)")
      ("output-steps", po::value<int>()->default_value(50), "Number of output steps")
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

std::string asciiLogo()
{
  return std::string("\n\033[38;2;142;232;216m") + logo + "\033[0m\n" + "Focused Ultrasounds High Order Matrix Free FEM GPU solver for unstructured meshes\n";
}
