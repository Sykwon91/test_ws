#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <Eigen/Dense>

// 1. 스캔 포인트에 R, t 적용
std::vector<Eigen::Vector3f> transformPointCloud(
    const std::vector<Eigen::Vector3f>& points,
    const Eigen::Matrix3f& R,
    const Eigen::Vector3f& t)
{
    std::vector<Eigen::Vector3f> out;
    out.reserve(points.size());

    for (const auto& p : points) {
        out.push_back(R * p + t);
    }
    return out;
}

// 2. 최근접점 매칭 (브루트포스)
//    src: 스캔(측정) 포인트들 (이미 변환된 상태)
//    dst: src 각 점에 대응되는 "맵 포인트"
void findCorrespondences(
    const std::vector<Eigen::Vector3f>& map_points,
    const std::vector<Eigen::Vector3f>& scan_points,
    std::vector<Eigen::Vector3f>& src,   // scan에서 온 점들
    std::vector<Eigen::Vector3f>& dst)   // map에서 온 점들
{
    src.clear();
    dst.clear();

    if (map_points.empty() || scan_points.empty())
        return;

    const float max_dist = 5.0f;            // 5m 넘으면 매칭 안 한다 (옵션)
    const float max_dist2 = max_dist * max_dist;

    for (const auto& sp : scan_points) {
        float best_d2 = std::numeric_limits<float>::max();
        int best_idx = -1;

        for (size_t j = 0; j < map_points.size(); ++j) {
            Eigen::Vector3f diff = sp - map_points[j];
            float d2 = diff.squaredNorm();
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = static_cast<int>(j);
            }
        }

        if (best_idx >= 0 && best_d2 < max_dist2) {
            src.push_back(sp);
            dst.push_back(map_points[best_idx]);
        }
    }
}

// 3. 매칭된 점 쌍(src ↔ dst)에서 R, t 추정 (SVD)
bool estimateRigidTransform(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst,
    Eigen::Matrix3f& R,
    Eigen::Vector3f& t)
{
    if (src.size() < 3 || src.size() != dst.size())
        return false;

    // 3-1. 각 집합의 centroid
    Eigen::Vector3f src_centroid = Eigen::Vector3f::Zero();
    Eigen::Vector3f dst_centroid = Eigen::Vector3f::Zero();

    for (size_t i = 0; i < src.size(); ++i) {
        src_centroid += src[i];
        dst_centroid += dst[i];
    }
    src_centroid /= static_cast<float>(src.size());
    dst_centroid /= static_cast<float>(dst.size());

    // 3-2. 중심 기준으로 빼고 공분산 행렬 H 계산
    Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
    for (size_t i = 0; i < src.size(); ++i) {
        Eigen::Vector3f ps = src[i] - src_centroid;
        Eigen::Vector3f pd = dst[i] - dst_centroid;
        H += ps * pd.transpose();
    }

    // 3-3. SVD
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Matrix3f V = svd.matrixV();

    Eigen::Matrix3f R_est = V * U.transpose();

    // 반사(reflection) 방지 (det(R)=+1 유지)
    if (R_est.determinant() < 0) {
        V.col(2) *= -1.0f;
        R_est = V * U.transpose();
    }

    Eigen::Vector3f t_est = dst_centroid - R_est * src_centroid;

    R = R_est;
    t = t_est;

    return true;
}

// 4. RMS 에러 계산 (디버깅용)
double computeRMSError(
    const std::vector<Eigen::Vector3f>& src,
    const std::vector<Eigen::Vector3f>& dst)
{
    if (src.empty() || src.size() != dst.size())
        return 0.0;

    double sum = 0.0;
    for (size_t i = 0; i < src.size(); ++i) {
        Eigen::Vector3f diff = src[i] - dst[i];
        sum += diff.squaredNorm();
    }
    double mse = sum / static_cast<double>(src.size());
    return std::sqrt(mse);
}

int main()
{
    // === 예시 맵 포인트 (월드 좌표계) ===
    std::vector<Eigen::Vector3f> map_points;
    map_points.push_back(Eigen::Vector3f(1.0f, 1.0f, 0.0f));
    map_points.push_back(Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    map_points.push_back(Eigen::Vector3f(2.0f, 3.0f, 0.5f));
    map_points.push_back(Eigen::Vector3f(3.0f, 2.0f, 0.2f));

    // === 로봇의 "진짜" 포즈 (테스트용 ground truth) ===
    float theta_true = M_PI / 4.0f;   // 45deg 회전
    Eigen::Matrix3f R_true;
    R_true = Eigen::AngleAxisf(theta_true, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(theta_true, Eigen::Vector3f::UnitY());
    Eigen::Vector3f t_true(1.0f, -0.5f, 0.0f);

    // === 로봇이 측정한 로컬 포인트 (맵 포인트를 로봇 좌표로 본다고 가정) ===
    //     실제로는 LiDAR 스캔이지만, 여기선 테스트용으로 맵을 역변환해서 만듦
    std::vector<Eigen::Vector3f> local_points;
    local_points.reserve(map_points.size());
    Eigen::Matrix3f R_true_inv = R_true.transpose();
    for (const auto& P : map_points) {
        // P = R_true * p_local + t_true  →  p_local = R_true^T * (P - t_true)
        Eigen::Vector3f p_local = R_true_inv * (P - t_true);
        local_points.push_back(p_local);
    }

    // === ICP로 포즈 추정 시작 ===
    Eigen::Matrix3f R_est = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t_est = Eigen::Vector3f::Zero();

    for (int iter = 0; iter < 10; ++iter) {
        // 1) 현재 추정 포즈로 로컬 포인트를 월드로 변환
        auto scan_in_world = transformPointCloud(local_points, R_est, t_est);

        // 2) 맵과 최근접 매칭
        std::vector<Eigen::Vector3f> src, dst;
        findCorrespondences(map_points, scan_in_world, src, dst);

        if (src.size() < 3) {
            std::cout << "Not enough correspondences" << std::endl;
            break;
        }

        // 3) 매칭된 점 쌍으로부터 delta R, delta t 추정
        Eigen::Matrix3f dR;
        Eigen::Vector3f dt;
        if (!estimateRigidTransform(src, dst, dR, dt)) {
            std::cout << "estimateRigidTransform failed" << std::endl;
            break;
        }

        // 4) 포즈 업데이트: T_new = (dR, dt) * (R_est, t_est)
        t_est = dR * t_est + dt;
        R_est = dR * R_est;

        double rms = computeRMSError(src, dst);
        std::cout << "Iter " << iter << ", RMS = " << rms << std::endl;
        std::cout << "\n=== Estimated pose ===\n";
        std::cout << "R_est:\n" << R_est << std::endl;
        std::cout << "t_est: " << t_est.transpose() << std::endl;
    }

    std::cout << "\n=== True pose ===\n";
    std::cout << "R_true:\n" << R_true << std::endl;
    std::cout << "t_true: " << t_true.transpose() << std::endl;



    return 0;
}
