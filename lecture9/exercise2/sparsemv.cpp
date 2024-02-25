#include <iostream>
#include <Kokkos_Core.hpp>
#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spmv.hpp"

int main(int argc, char *argv[])
{
    Kokkos::initialize(argc, argv);
    {
        int n = 1 << 27;
        int num_rows = n;
        int num_cols = n;
        int nnz = 3 * n - 2;

        Kokkos::View<int *> row_ptr("row_ptr", num_rows + 1);
        Kokkos::View<int *> col_ind("col_ind", nnz);
        Kokkos::View<double *> values("values", nnz);

        // create a simple tridiagonal matrix 1 -2 1
        Kokkos::parallel_for(
            "fill_crs", num_rows, KOKKOS_LAMBDA(const int i) {
                if (i == 0)
                {
                    row_ptr(0) = 0;
                    col_ind(0) = 0;
                    col_ind(1) = 1;
                    values(0) = -2.0;
                    values(1) = 1.0;
                }
                else if (i < num_rows - 1)
                {
                    row_ptr(i) = 3 * i - 1;
                    col_ind(3 * i - 1) = i - 1;
                    col_ind(3 * i) = i;
                    col_ind(3 * i + 1) = i + 1;
                    values(3 * i - 1) = 1.0;
                    values(3 * i) = -2.0;
                    values(3 * i + 1) = 1.0;
                }
                else
                {
                    row_ptr(i) = 3 * i - 1;
                    col_ind(3 * i - 1) = i - 1;
                    col_ind(3 * i) = i;
                    values(3 * i - 1) = 1.0;
                    values(3 * i) = -2.0;
                }
                row_ptr(num_rows) = 3 * num_rows - 2;
            });

        Kokkos::fence();

        KokkosSparse::CrsMatrix<double, int, Kokkos::DefaultExecutionSpace, void, int> A("A", num_rows, num_cols, nnz, values, row_ptr, col_ind);

        Kokkos::View<double *> x("x", num_cols);
        Kokkos::View<double *> y("y", num_rows);
        Kokkos::deep_copy(x, 1.0);
        Kokkos::deep_copy(y, 0.0);

        KokkosSparse::spmv("N", 1.0, A, x, 0.0, y);

        auto y_h = Kokkos::create_mirror_view(y);
        Kokkos::deep_copy(y_h, y);

        for (int i = 0; i < num_rows; i++)
        {
            std::cout << y_h(i) << " ";
        }
        std::cout << std::endl;

        Kokkos::fence();
    }

    Kokkos::finalize();
    return 0;
}
