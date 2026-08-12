#include "convert.cpp"

using matrixd = std::vector<std::vector<double>>;
using vectord = std::vector<double>;

class PCAAnomalyDetector
{
private:
    int block_size;
    int k; // components
    int D; // feature vector length
    vectord mean;
    matrixd W_k;
    bool is_trained = false;

public:
    PCAAnomalyDetector(int b_size, int components)
        : block_size(b_size), k(components), D(b_size * b_size * 3) {}

    matrixd extractBlocks(const Image &img, int &out_h, int &out_w)
    {
        out_h = img.height - (img.height % block_size);
        out_w = img.width - (img.width % block_size);

        int num_blocks = (out_h / block_size) * (out_w / block_size);
        int patch_pixels = block_size * block_size;

        matrixd X(num_blocks, vectord(D, 0.0));
        int block_idx = 0;

        for (int r = 0; r < out_h; r += block_size)
        {
            for (int c = 0; c < out_w; c += block_size)
            {
                int pixel_index = 0;
                for (int i = 0; i < block_size; i++)
                {
                    for (int j = 0; j < block_size; j++)
                    {
                        X[block_idx][pixel_index] = img.r[r + i][c + j];
                        X[block_idx][pixel_index + patch_pixels] = img.g[r + i][c + j];
                        X[block_idx][pixel_index + patch_pixels * 2] = img.b[r + i][c + j];
                        pixel_index++;
                    }
                }
                block_idx++;
            }
        }
        return X;
    }

    void train(const matrixd &X)
    {
        size_t m = X.size();

        // Mean centring
        mean.assign(D, 0.0);
        for (size_t i = 0; i < m; i++)
        {
            for (size_t j = 0; j < D; j++)
            {
                mean[j] += X[i][j];
            }
        }
        for (size_t i = 0; i < D; i++)
        {
            mean[i] /= m;
        }

        matrixd X_c(m, vectord(D, 0.0));
        for (size_t i = 0; i < m; ++i)
        {
            for (int j = 0; j < D; ++j)
                X_c[i][j] = X[i][j] - mean[j];
        }

        // Covariance matrix [192 * 192] (8 * 8 * 3)
        matrixd covariance(D, vectord(D, 0));
        for (size_t i = 0; i < D; i++)
        {
            for (size_t j = 0; j < D; j++)
            {
                double sum = 0.0;
                for (size_t r = 0; r < m; r++)
                    sum += X_c[r][i] * X_c[r][j];
                covariance[i][j] = sum / (m - 1);
            }
        }

        W_k.assign(k, vectord(D, 0.0));
        matrixd cov_def = covariance;

        for (int c = 0; c < k; c++)
        {
            vectord v(D, 1.0);
            for (int iter = 0; iter < 100; iter++)
            {
                vectord v_next(D, 0.0);
                for (int i = 0; i < D; i++)
                {
                    for (int j = 0; j < D; j++)
                    {
                        v_next[i] += cov_def[i][j] * v[j];
                    }
                }
                double normal = 0.0;

                for (double val : v_next)
                {
                    normal += val * val;
                }
                normal = std::sqrt(normal);
                if (normal < 1e-12)
                {
                    break;
                }
                for (int i = 0; i < D; i++)
                {
                    v[i] = v_next[i] / normal;
                }
            }

            double lambda = 0.0f;
            for (size_t i = 0; i < D; i++)
            {
                double temp = 0.0f;
                for (int j = 0; j < D; j++)
                {
                    temp += cov_def[i][j] * v[j];
                }
                lambda += v[i] * temp;
            }

            W_k[c] = v;
            for (int i = 0; i < D; i++)
            {
                for (int j = 0; j < D; j++)
                {
                    cov_def[i][j] -= lambda * v[i] * v[j];
                }
            }
        }
        is_trained = true;
    }

    void inspect(const matrixd &X, int h, int w, double sensitivity,
                 matrixd &r_error, matrixd &g_error, matrixd &b_error) const
    {
        r_error.assign(h, vectord(w, 0));
        g_error.assign(h, vectord(w, 0));
        b_error.assign(h, vectord(w, 0));

        int blocks_per_row = w / block_size;
        int pixels_per_channel = block_size * block_size;
        int num_patches = X.size();

#pragma omp parallel for
        for (int patch_index = 0; patch_index < num_patches; patch_index++)
        {
            int r = (patch_index / blocks_per_row) * block_size;
            int c = (patch_index % blocks_per_row) * block_size;

            // project test patch in clean basis: z = (x - mean) * W_k^T
            vectord z(k, 0.0);
            for (int comp = 0; comp < k; comp++)
            {
                double proj = 0.0;
                for (int j = 0; j < D; j++)
                {
                    proj += (X[patch_index][j] - mean[j]) * W_k[comp][j];
                }
                z[comp] = proj;
            }

            // reconstruct patch: x_^ = z * W_k + mean
            vectord x_hat(D, 0.0);
            for (int j = 0; j < D; j++)
            {
                double val = 0.0;
                for (int comp = 0; comp < k; comp++)
                {
                    val += z[comp] * W_k[comp][j];
                }
                x_hat[j] = val + mean[j];
            }

            int pixel_sub_index = 0;
            for (int i = 0; i < block_size; i++)
            {
                for (int j = 0; j < block_size; j++)
                {
                    double diff_r = std::abs(X[patch_index][pixel_sub_index] - x_hat[pixel_sub_index]);
                    double diff_g = std::abs(X[patch_index][pixel_sub_index + pixels_per_channel] - x_hat[pixel_sub_index + pixels_per_channel]);
                    double diff_b = std::abs(X[patch_index][pixel_sub_index + 2 * pixels_per_channel] - x_hat[pixel_sub_index + 2 * pixels_per_channel]);

                    double total_err = (diff_r + diff_g + diff_b) / 3.0;
                    double heatmap_intensity = total_err * sensitivity;

                    r_error[r + i][c + j] = heatmap_intensity * 2.0;
                    g_error[r + i][c + j] = heatmap_intensity * 0.2;
                    b_error[r + i][c + j] = heatmap_intensity * 0.2;

                    pixel_sub_index++;
                }
            }
        }
    }
};
