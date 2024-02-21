#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int n = 1 << 20;
        Kokkos::View<double *> d_x("x", n), d_y("y", n), d_z("z", n);
        auto h_z = Kokkos::create_mirror_view(d_z);

        Kokkos::parallel_for(
            "fill_vec", n, KOKKOS_LAMBDA(const int i) {
                d_x[i] = 1.0;
                d_y[i] = 2.0;
            });

        Kokkos::parallel_for(
            "vecadd", n, KOKKOS_LAMBDA(int i) {
                d_z(i) = d_x(i) + d_y(i);
            });

        Kokkos::fence();
        Kokkos::deep_copy(h_z, d_z);

        for (int i = 0; i < n; i++)
        {
            std::cout << h_z[i] << " ";
        }
        std::cout << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
