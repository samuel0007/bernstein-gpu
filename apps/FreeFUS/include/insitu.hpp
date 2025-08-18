#pragma once
#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <dolfinx.h>
#include <memory>

namespace freefus {

template <typename T>
void setup_insitu(std::shared_ptr<fem::FunctionSpace<T>> &V,
                  int polynomial_degree,
                  const std::shared_ptr<fem::Function<T>> &solution,
                  ascent::Ascent &ascent_runner, conduit::Node &conduit_mesh,
                  conduit::Node &ascent_actions, const UserConfig<T> &config,
                  const std::string &field_name = "u") {
  const int tdim = V->mesh()->topology()->dim();
  ascent_h::MeshToBlueprintMesh(V, polynomial_degree, conduit_mesh);
  ascent_h::FunctionToBlueprintField(solution, conduit_mesh, field_name);
  // ascent_h::MeshToBlueprintMesh<T>(V, conduit_mesh);
  // ascent_h::CG1FunctionToBlueprintField(solution, conduit_mesh, field_name);

  // ascent_h::FunctionToBlueprintField(solution, conduit_mesh, field_name);
  // Data is passed by reference
  // ascent_h::DG0FunctionToBlueprintField(solution, conduit_mesh, field_name);

  conduit::Node ascent_opts;

  ascent_opts["default_dir"] = config.mesh_dir;
  // ascent_opts["mpi_comm"] = MPI_Comm_c2f(V->mesh()->comm());
  ascent_runner.open(ascent_opts);

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

    scenes["s3/plots/p1/type"] = "pseudocolor";
    scenes["s3/plots/p1/field"] = field_name;
    scenes["s3/plots/p1/pipeline"] = "pl4";

    // scenes["s1/plots/p2/type"] = "mesh";
    scenes["s1/image_prefix"] = field_name;

    conduit::Node pipelines;
    pipelines["pl1/f1/type"] = "slice";
    pipelines["pl2/f1/type"] = "slice";
    pipelines["pl3/f1/type"] = "slice";

    // pipelines["pl4/f1/type"] = "clip";

    pipelines["pl4/f1/type"] = "contour";
    pipelines["pl4/f1/params/field"] = field_name;
    pipelines["pl4/f1/params/iso_values"] = {-10000, 10000};

    // {
    //   auto &params = pipelines["pl4/f1/params"];
    //   auto &point = params["point"];
    //   point["x_offset"] = 0.0;
    //   point["y_offset"] = 0.0;
    //   point["z_offset"] = 0.0;

    //   auto &normal = params["normal"];
    //   normal["x"] = 0.0;
    //   normal["y"] = 1.0;
    //   normal["z"] = 0.0;
    // }

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
    // scenes["s2/renders/r1/camera/azimuth"] = 45.0;
    // scenes["s2/renders/r1/camera/elevation"] = 40.0;

    scenes["s3/renders/r1/image_prefix"] = "ri1";
    scenes["s3/renders/r1/camera/azimuth"] = 20.0;
    scenes["s3/renders/r1/camera/elevation"] = 40.0;

    conduit::Node &add_pipelines = ascent_actions.append();
    add_pipelines["action"] = "add_pipelines";
    add_pipelines["pipelines"] = pipelines;

    // conduit::Node &add_extracts = ascent_actions.append();
    // add_extracts["action"] = "add_extracts";
    // conduit::Node &extracts = add_extracts["extracts"];
    // extracts["e1/type"]  = "relay";
    // extracts["e1/params/protocol"] = "hdf5";
    // extracts["e1/params/path"] = config.mesh_dir;
    // extracts["e1/params/num_files"] = 1;

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

  if (!config.insitu_with_yaml) {
    spdlog::info("Ascent actions {}", ascent_actions.to_yaml());
  }
}

void insitu_output_DG(auto material_coefficients,
                      ascent::Ascent &ascent_runner) {
  auto [c0, rho0, delta0, b0] = material_coefficients;
  conduit::Node conduit_mesh;
  conduit::Node ascent_actions;

  ascent_h::MeshToBlueprintMesh(c0->function_space(), conduit_mesh);
  ascent_h::DG0FunctionToBlueprintField(c0, conduit_mesh, "c0");
  ascent_h::DG0FunctionToBlueprintField(rho0, conduit_mesh, "rho0");
  ascent_h::DG0FunctionToBlueprintField(delta0, conduit_mesh, "delta0");
  ascent_h::DG0FunctionToBlueprintField(b0, conduit_mesh, "b0");

  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "slice";
  pipelines["pl2/f1/type"] = "slice";
  pipelines["pl3/f1/type"] = "slice";

  // pipelines["pl4/f1/type"] = "clip_with_field";

  // pipelines["pl4/f1/type"] = "contour";
  // pipelines["pl4/f1/params/iso_values"] = 1850.;
  // pipelines["pl4/f1/params/field"] = "rho0";
  // pipelines["pl4/f1/params/clip_value"] = 1850.;
  // pipelines["pl4/f1/params/invert"] = "true";


  // {
  //   auto &params = pipelines["pl4/f1/params"];
  //   auto &point = params["point"];
  //   point["x_offset"] = 0.0;
  //   point["y_offset"] = 0.0;
  //   point["z_offset"] = 0.0;

  //   auto &normal = params["normal"];
  //   normal["x"] = 0.0;
  //   normal["y"] = 1.0;
  //   normal["z"] = 0.0;
  // }

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

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = "c0";
  scenes["s1/renders/r1/image_prefix"] = "materialc0";
  scenes["s1/renders/r1/camera/azimuth"] = 20.0;
  scenes["s1/renders/r1/camera/elevation"] = 40.0;

  double vec3[3];
  vec3[0] = 1.; vec3[1] = 0.; vec3[2] = 0.;
  // scenes["s2/plots/p4/type"] = "volume";
  // scenes["s2/plots/p4/field"] = "rho0";
  // scenes["s2/plots/p4/color_table/annotation"] = "false";
  scenes["s2/renders/r1/image_prefix"] = "materialrho01";
  scenes["s2/renders/r1/camera/azimuth"] = 20.0;
  scenes["s2/renders/r1/camera/elevation"] = 10.0;
  scenes["s2/renders/r1/camera/up"].set_float64_ptr(vec3,3);
  // scenes["s2/renders/r1/world_annotations"] = "false";
  // scenes["s2/renders/r1/screen_annotations"] = "false";

  scenes["s2/plots/p1/type"] = "pseudocolor";
  scenes["s2/plots/p1/field"] = "rho0";
  scenes["s2/plots/p1/pipeline"] = "pl4";
  scenes["s2/plots/p1/type"] = "pseudocolor";
  scenes["s2/plots/p1/field"] = "rho0";
  scenes["s2/plots/p1/pipeline"] = "pl1";
  scenes["s2/plots/p2/type"] = "pseudocolor";
  scenes["s2/plots/p2/field"] = "rho0";
  scenes["s2/plots/p2/pipeline"] = "pl2";
  scenes["s2/plots/p2/color_table/annotation"] = "false";
  scenes["s2/plots/p3/type"] = "pseudocolor";
  scenes["s2/plots/p3/field"] = "rho0";
  scenes["s2/plots/p3/pipeline"] = "pl3";
  scenes["s2/plots/p3/color_table/annotation"] = "false";

  scenes["s3/plots/p1/type"] = "pseudocolor";
  scenes["s3/plots/p1/field"] = "delta0";
  scenes["s3/renders/r1/image_prefix"] = "materialdelta0";
  scenes["s3/renders/r1/camera/azimuth"] = 180.0;
  scenes["s3/renders/r1/camera/elevation"] = 40.0;
  scenes["s3/renders/r1/camera/up"].set_float64_ptr(vec3,3);


  scenes["s4/plots/p1/type"] = "pseudocolor";
  scenes["s4/plots/p1/field"] = "b0";
  scenes["s4/renders/r1/image_prefix"] = "materialb0";
  scenes["s4/renders/r1/camera/azimuth"] = 20.0;
  scenes["s4/renders/r1/camera/elevation"] = 40.0;
  scenes["s4/renders/r1/camera/up"].set_float64_ptr(vec3,3);


  conduit::Node &add_scenes = ascent_actions.append();
  add_scenes["action"] = "add_scenes";
  add_scenes["scenes"] = scenes;

  conduit::Node &add_pipelines = ascent_actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;

  ascent_runner.publish(conduit_mesh);
  ascent_runner.execute(ascent_actions);
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
