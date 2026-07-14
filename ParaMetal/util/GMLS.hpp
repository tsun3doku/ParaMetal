#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <glm/glm.hpp>

namespace GMLS {

inline constexpr double RankTolerance = 1e-12;
inline constexpr double ReproductionTolerance = 1e-9;

inline double wendlandC2(double normalizedRadius) {
    if (normalizedRadius >= 1.0) {
        return 0.0;
    }
    const double oneMinusRadius = 1.0 - std::max(0.0, normalizedRadius);
    return oneMinusRadius * oneMinusRadius * oneMinusRadius * oneMinusRadius *
        ((4.0 * normalizedRadius) + 1.0);
}

inline bool solveWeights(
    const Eigen::MatrixXd& basis,
    const Eigen::VectorXd& kernelWeights,
    const std::vector<Eigen::VectorXd>& functionals,
    std::vector<Eigen::VectorXd>& weights) {
    if (basis.rows() == 0 || basis.rows() != kernelWeights.size()) {
        return false;
    }

    Eigen::MatrixXd weightedBasis = basis;
    for (Eigen::Index row = 0; row < weightedBasis.rows(); ++row) {
        weightedBasis.row(row) *= std::sqrt(kernelWeights(row));
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(weightedBasis, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(RankTolerance);
    if (svd.rank() != basis.cols()) {
        return false;
    }

    const Eigen::VectorXd singularValues = svd.singularValues();
    weights.clear();
    for (const Eigen::VectorXd& functional : functionals) {
        Eigen::VectorXd spectral = svd.matrixV().transpose() * functional;
        for (Eigen::Index index = 0; index < spectral.size(); ++index) {
            spectral(index) /= singularValues(index);
        }

        Eigen::VectorXd result = svd.matrixU() * spectral;
        for (Eigen::Index row = 0; row < result.size(); ++row) {
            result(row) *= std::sqrt(kernelWeights(row));
        }
        if (!result.allFinite() ||
            (basis.transpose() * result - functional).cwiseAbs().maxCoeff() > ReproductionTolerance) {
            return false;
        }
        weights.push_back(std::move(result));
    }
    return true;
}

inline bool computeSurfaceWeights(
    const glm::dvec3& targetPosition,
    const glm::dvec3& targetNormal,
    const std::vector<glm::dvec3>& sourcePositions,
    double kernelRadius,
    std::vector<double>& valueWeights,
    std::vector<glm::dvec3>& gradientWeights) {
    valueWeights.clear();
    gradientWeights.clear();
    if (sourcePositions.size() < 3 || kernelRadius <= 0.0) {
        return false;
    }

    const double normalLength = glm::length(targetNormal);
    if (!std::isfinite(normalLength) || normalLength <= 1e-12) {
        return false;
    }
    const glm::dvec3 normal = targetNormal / normalLength;
    glm::dvec3 referenceAxis;
    if (std::abs(normal.x) <= std::abs(normal.y) && std::abs(normal.x) <= std::abs(normal.z)) {
        referenceAxis = glm::dvec3(1.0, 0.0, 0.0);
    } else if (std::abs(normal.y) <= std::abs(normal.z)) {
        referenceAxis = glm::dvec3(0.0, 1.0, 0.0);
    } else {
        referenceAxis = glm::dvec3(0.0, 0.0, 1.0);
    }
    const glm::dvec3 tangentU = glm::normalize(glm::cross(normal, referenceAxis));
    const glm::dvec3 tangentV = glm::cross(normal, tangentU);

    Eigen::MatrixXd basis(sourcePositions.size(), 3);
    Eigen::VectorXd kernelWeights(sourcePositions.size());
    for (size_t index = 0; index < sourcePositions.size(); ++index) {
        const glm::dvec3 delta = sourcePositions[index] - targetPosition;
        const double u = glm::dot(delta, tangentU) / kernelRadius;
        const double v = glm::dot(delta, tangentV) / kernelRadius;
        basis.row(index) << 1.0, u, v;
        kernelWeights(index) = wendlandC2(std::sqrt(u * u + v * v));
    }

    std::vector<Eigen::VectorXd> functionals(3, Eigen::VectorXd::Zero(3));
    functionals[0](0) = 1.0;
    functionals[1](1) = 1.0;
    functionals[2](2) = 1.0;
    std::vector<Eigen::VectorXd> solvedWeights;
    if (!solveWeights(basis, kernelWeights, functionals, solvedWeights)) {
        return false;
    }

    valueWeights.resize(sourcePositions.size());
    gradientWeights.resize(sourcePositions.size());
    for (size_t index = 0; index < sourcePositions.size(); ++index) {
        valueWeights[index] = solvedWeights[0](index);
        gradientWeights[index] =
            (solvedWeights[1](index) * tangentU + solvedWeights[2](index) * tangentV) / kernelRadius;
    }
    return true;
}

inline bool computeContactValueWeights(
    const glm::dvec3& targetPosition,
    const std::vector<glm::dvec3>& sourcePositions,
    double kernelRadius,
    std::vector<double>& valueWeights) {
    valueWeights.clear();
    if (sourcePositions.size() < 4 || kernelRadius <= 0.0) {
        return false;
    }

    Eigen::MatrixXd basis(sourcePositions.size(), 4);
    Eigen::VectorXd kernelWeights(sourcePositions.size());
    for (size_t index = 0; index < sourcePositions.size(); ++index) {
        const glm::dvec3 delta = (sourcePositions[index] - targetPosition) / kernelRadius;
        basis.row(index) << 1.0, delta.x, delta.y, delta.z;
        kernelWeights(index) = wendlandC2(glm::length(delta));
    }

    std::vector<Eigen::VectorXd> functionals(1, Eigen::VectorXd::Zero(4));
    functionals[0](0) = 1.0;
    std::vector<Eigen::VectorXd> solvedWeights;
    if (!solveWeights(basis, kernelWeights, functionals, solvedWeights)) {
        return false;
    }
    valueWeights.resize(sourcePositions.size());
    for (size_t index = 0; index < sourcePositions.size(); ++index) {
        valueWeights[index] = solvedWeights[0](index);
    }
    return true;
}

} // namespace GMLS
