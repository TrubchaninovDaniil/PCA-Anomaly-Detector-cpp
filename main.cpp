#include <cmath>
#include <omp.h>
#include "pca_anomaly_detector.cpp"

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " <clean_ref.jpg> <test_part.jpg> <output_heatmap.png> [k_components] [sensitivity]\n";
        std::cout << "Example: " << argv[0] << " pristine_pcb.jpg candidate_pcb.jpg heatmap.jpg 8 5.0\n";
        return 1;
    }

    std::string ref_path = argv[1];
    std::string test_path = argv[2];
    std::string out_path = argv[3];
    int k_components = (argc > 4) ? std::stoi(argv[4]) : 8;
    double sensitivity = (argc > 5) ? std::stof(argv[5]) : 5.0f;

    std::cout << "Loading pristine reference sample:" << ref_path << "...\n";
    Image reference_img = Image::loadImage(ref_path);

    PCAAnomalyDetector detector(8, k_components);

    int reference_h, reference_w;
    auto X_clean = detector.extractBlocks(reference_img, reference_h, reference_w);

    std::cout << "Training PCA basis k=" << k_components << "\n";
    detector.train(X_clean);

    std::cout << "Loading candidate sample: " << test_path << "...\n";
    Image test_img = Image::loadImage(test_path);

    int test_h, test_w;
    auto X_test = detector.extractBlocks(test_img, test_h, test_w);

    std::cout << "Computing reconstruction error map: \n";
    std::vector<std::vector<double>> error_r, error_g, error_b;

    Image::saveJPG(out_path, error_r, error_g, error_b);
    std::cout << "Defect heatmap saved to: " << out_path << "\n";
}