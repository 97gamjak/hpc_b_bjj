#include <KokkosBlas.hpp>
#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int n = 1 << 28;
        Kokkos::View<double*, Kokkos::LayoutRight> d_x("x", n), d_y("y", n);
        // auto h_z = Kokkos::create_mirror_view(d_z);
        auto md_range = Kokkos::RangePolicy(0, n);

        Kokkos::parallel_for(
            "fill_vec", md_range, KOKKOS_LAMBDA(const int i) {
                d_x(i) = 1.0;
                d_y(i) = 2.0;
            });

        Kokkos::fence();

        double inner1 = 0.0;
        Kokkos::parallel_reduce(
            "dot_product", n,
            KOKKOS_LAMBDA(int i, double& update) {
                update += d_x(i) * d_y(i);
            },
            inner1);

        double inner2 = KokkosBlas::dot(d_x, d_y);

        Kokkos::fence();
    }
    Kokkos::finalize();
    return 0;
}