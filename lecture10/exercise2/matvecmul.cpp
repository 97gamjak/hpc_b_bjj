#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp>

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int size = 100;

        Kokkos::View<double **> A("A", size, size);
        Kokkos::View<double *> x("x", size);
        Kokkos::View<double *> y_1("y_1", size);
        Kokkos::View<double *> y_2("y_2", size);

        Kokkos::parallel_for(
            "InitA", size, KOKKOS_LAMBDA(const int i) {
                for (int j = 0; j < size; j++)
                {
                    A(i, j) = 1.0 * i + j;
                }
            });

        Kokkos::parallel_for(
            "InitX", size, KOKKOS_LAMBDA(const int i) {
                x(i) = 1.0 * i;
            });

        Kokkos::parallel_for(
            "MatVecProduct", size, KOKKOS_LAMBDA(const int i) {
                double sum = 0;
                for (int j = 0; j < size; j++)
                {
                    sum += A(i, j) * x(j);
                }
                y_1(i) = sum;
            });

        KokkosBlas::gemv("N", 1.0, A, x, 0.0, y_2);

        Kokkos::fence();
    }
    Kokkos::finalize();
    return 0;
}