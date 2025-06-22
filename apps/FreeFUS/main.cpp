#include "ascent_helpers.hpp"
#include "cli.hpp"
#include "vector.hpp"

#include <spdlog/spdlog.h>

using namespace dolfinx;
using T = SCALAR_TYPE;

int main(int argc, char *argv[])
{
    auto vm = get_cli_config<T>(argc, argv);
    if (vm.count("help"))
    {
        std::cout << "...";
        return 0;
    }

    spdlog::info(asciiLogo());
    UserConfig<T> config = make_user_config<T>(vm);
    display_user_config(config);


    return 0;
}