#pragma once
#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <dolfinx.h>
#include <memory>

namespace freefus {

template <typename T>
void setup_insitu(const std::shared_ptr<fem::FunctionSpace<T>> &V,
                  int polynomial_degree,
                  const std::shared_ptr<fem::Function<T>> &solution,
                  ascent::Ascent &ascent_runner, conduit::Node &conduit_mesh,
                  conduit::Node &ascent_actions,
                  const std::string &field_name = "u") {
  const int tdim = V->mesh()->topology()->dim();
  ascent_h::MeshToBlueprintMesh(V, polynomial_degree, conduit_mesh);
  // Data is passed by reference
  ascent_h::FunctionToBlueprintField(solution, conduit_mesh, field_name);

  conduit::Node ascent_opts;
  // ascent_opts["mpi_comm"] = MPI_Comm_c2f(V->mesh()->comm());
  // ascent_runner.open(ascent_opts);
  ascent_runner.open();

  conduit::Node scenes;
  if (tdim == 3) {

    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = field_name;
    scenes["s1/plots/p1/pipeline"] = "pl1";

    scenes["s1/plots/p2/type"] = "pseudocolor";
    scenes["s1/plots/p2/field"] = field_name;
    scenes["s1/plots/p2/pipeline"] = "pl2";
    scenes["s1/plots/p2/color_table/annotation"] = "false";

    scenes["s1/plots/p3/type"] = "pseudocolor";
    scenes["s1/plots/p3/field"] = field_name;
    scenes["s1/plots/p3/pipeline"] = "pl3";
    scenes["s1/plots/p3/color_table/annotation"] = "false";

    // scenes["s2/plots/p1/type"] = "volume";
    // scenes["s2/plots/p1/field"] = field_name;

    // scenes["s1/plots/p2/type"] = "mesh";
    scenes["s1/image_prefix"] = field_name;

    conduit::Node pipelines;
    pipelines["pl1/f1/type"] = "slice";
    pipelines["pl2/f1/type"] = "slice";
    pipelines["pl3/f1/type"] = "slice";

    {
      auto &params = pipelines["pl1/f1/params"];
      auto &point = params["point"];
      point["x_offset"] = 0.0;
      point["y_offset"] = 0.0;
      point["z_offset"] = 0.0;

      auto &normal = params["normal"];
      normal["x"] = 0.0;
      normal["y"] = 1.0;
      normal["z"] = 0.0;
    }

    {
      auto &params = pipelines["pl2/f1/params"];
      auto &point = params["point"];
      point["x_offset"] = 0.0;
      point["y_offset"] = 0.0;
      point["z_offset"] = 0.0;

      auto &normal = params["normal"];
      normal["x"] = 0.0;
      normal["y"] = 0.0;
      normal["z"] = 1.0;
    }

    {
      auto &params = pipelines["pl3/f1/params"];
      auto &point = params["point"];
      point["x_offset"] = 0.0;
      point["y_offset"] = 0.0;
      point["z_offset"] = 0.0;

      auto &normal = params["normal"];
      normal["x"] = 1.0;
      normal["y"] = 0.0;
      normal["z"] = 0.0;
    }

    // common camera “look_at” and “up”

    // --- first render: high‐right‐front view ---
    // double pos1[3] = {1., 1., 1.};
    scenes["s1/renders/r1/image_prefix"] = "r1";
    scenes["s1/renders/r1/camera/azimuth"] = 20.0;
    scenes["s1/renders/r1/camera/elevation"] = 40.0;

    scenes["s1/renders/r2/image_prefix"] = "r2";
    scenes["s1/renders/r2/camera/azimuth"] = 10.0;
    scenes["s1/renders/r2/camera/elevation"] = 80.0;

    // scenes["s2/renders/r1/image_prefix"] = "rv1";
    // scenes["s2/renders/r1/camera/azimuth"] = 20.0;
    // scenes["s2/renders/r1/camera/elevation"] = 40.0;

    // scenes["s2/renders/r2/image_prefix"] = "rv2";
    // scenes["s2/renders/r2/camera/azimuth"] = -20.0;
    // scenes["s2/renders/r2/camera/elevation"] = 40.0;

    conduit::Node &add_pipelines = ascent_actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

  } else {
    scenes["s1/plots/p1/type"] = "pseudocolor";
    scenes["s1/plots/p1/field"] = field_name;
    // scenes["s1/plots/p2/type"] = "mesh";
    scenes["s1/renders/r1/camera/azimuth"] = 0.0;
    scenes["s1/renders/r1/image_prefix"] = "r1";
  }

  conduit::Node &add_scenes = ascent_actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  spdlog::info("Ascent actions {}", ascent_actions.to_yaml());
}

template <typename T>
void publish_insitu(const std::shared_ptr<fem::Function<T>> &solution,
                    ascent::Ascent &ascent_runner, conduit::Node &conduit_mesh,
                    conduit::Node &ascent_actions,
                    const std::string &field_name = "u") {
  ascent_runner.publish(conduit_mesh);
  ascent_runner.execute(ascent_actions);
}

} // namespace freefus
