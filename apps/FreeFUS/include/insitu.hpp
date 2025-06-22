#pragma once
#include <ascent.hpp>
#include <conduit_blueprint.hpp>
#include <dolfinx.h>
#include <memory>

template <typename T>
void setup_insitu(const std::shared_ptr<fem::FunctionSpace<T>> &V,
                  int polynomial_degree,
                  const std::shared_ptr<fem::Function<T>> &solution,
                  ascent::Ascent &ascent_runner, conduit::Node &conduit_mesh,
                  conduit::Node &ascent_actions,
                  const std::string &field_name = "u") {
  ascent_h::MeshToBlueprintMesh(V, polynomial_degree, conduit_mesh);
  // Data is passed by reference
  ascent_h::FunctionToBlueprintField(solution, conduit_mesh, field_name);

  ascent_runner.open();
  ascent_runner.publish(conduit_mesh);

  conduit::Node scenes;
  scenes["s1/plots/p1/type"] = "pseudocolor";
  scenes["s1/plots/p1/field"] = field_name;
  scenes["s1/plots/p2/type"] = "mesh";
  scenes["s1/image_prefix"] = field_name;

  conduit::Node &add_act = ascent_actions.append();
  add_act["action"] = "add_scenes";
  add_act["scenes"] = scenes;

  spdlog::info("Ascent actions {}", ascent_actions.to_yaml());
}