#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int n1 = 1 << 9, n2 = 1 << 9, n3 = 1 << 9;
        Kokkos::View<double ***, Kokkos::RightLayout> d_x("x", n1, n2, n3), d_y("y", n1, n2, n3), d_z("z", n1, n2, n3);
        auto h_z = Kokkos::create_mirror_view(d_z);
        auto md_range = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n1, n2, n3});

        Kokkos::parallel_for(
            "fill_vec", md_range, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                d_x(i, j, k) = 1.0;
                d_y(i, j, k) = 2.0;
            });

        Kokkos::parallel_for(
            "vecadd", md_range, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                d_z(i, j, k) = d_x(i, j, k) + d_y(i, j, k);
            });

        Kokkos::fence();
        Kokkos::deep_copy(h_z, d_z);
    }
    Kokkos::finalize();
    return 0;
}
