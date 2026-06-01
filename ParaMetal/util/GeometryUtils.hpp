#pragma once

#include <glm/glm.hpp>
#include <array>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

inline glm::vec3 safeNormalize(const glm::vec3& v) {
	const float lengthSquared = glm::dot(v, v);
	if (lengthSquared <= 1e-12f) {
		return glm::vec3(0.0f);
	}
	return v * (1.0f / std::sqrt(lengthSquared));
}

inline glm::mat4 toMat4(const std::array<float, 16>& values) {
	glm::mat4 matrix(1.0f);
	for (int col = 0; col < 4; ++col) {
		for (int row = 0; row < 4; ++row) {
			matrix[col][row] = values[static_cast<std::size_t>(col) * 4 + static_cast<std::size_t>(row)];
		}
	}
	return matrix;
}

inline std::array<float, 16> toMatrixArray(const glm::mat4& matrix) {
	return {
		matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],
		matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3],
		matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3],
		matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]
	};
}

inline bool intersectRayTriangle(
	const glm::vec3& orig, const glm::vec3& dir,
	const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
	float& outT, float& outU, float& outV) {

	const float EPS = 1e-8f;
	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;
	glm::vec3 pvec = glm::cross(dir, e2);
	float det = glm::dot(e1, pvec);

	if (std::abs(det) < EPS) {
		return false;
	}

	float invDet = 1.0f / det;
	glm::vec3 tvec = orig - v0;
	float u = glm::dot(tvec, pvec) * invDet;

	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	glm::vec3 qvec = glm::cross(tvec, e1);
	float v = glm::dot(dir, qvec) * invDet;

	if (v < 0.0f || (u + v) > 1.0f) {
		return false;
	}

	float t = glm::dot(e2, qvec) * invDet;
	if (t < 0.0f) {
		return false;
	}
	
	outT = t;
	outU = u;
	outV = v;
	
	return true;
}

inline glm::vec3 closestPointOnTriangle(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
	glm::vec3 ab = b - a;
	glm::vec3 ac = c - a;
	glm::vec3 ap = p - a;

	float d1 = glm::dot(ab, ap);
	float d2 = glm::dot(ac, ap);

	if (d1 <= 0.0f && d2 <= 0.0f) {
		return a;
	}

	glm::vec3 bp = p - b;
	float d3 = glm::dot(ab, bp);
	float d4 = glm::dot(ac, bp);

	if (d3 >= 0.0f && d4 <= d3) {
		return b;
	}

	float vc = d1 * d4 - d3 * d2;

	if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
		float v = d1 / (d1 - d3);
		return a + v * ab;
	}

	glm::vec3 cp = p - c;
	float d5 = glm::dot(ab, cp);
	float d6 = glm::dot(ac, cp);

	if (d6 >= 0.0f && d5 <= d6) {
		return c;
	}

	float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
		float w = d2 / (d2 - d6);
		return a + w * ac;
	}

	float va = d3 * d6 - d5 * d4;
	if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
		float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
		return b + w * (c - b);
	}

	float denom = 1.0f / (va + vb + vc);
	float v = vb * denom;
	float w = vc * denom;
	return a + ab * v + ac * w;
}

inline std::vector<float> computeVertexAreas(const std::vector<glm::vec3>& positions, const std::vector<uint32_t>& indices) {
	std::vector<float> vertexAreas(positions.size(), 0.0f);
	const size_t triangleCount = indices.size() / 3;

	for (size_t t = 0; t < triangleCount; ++t) {
		const uint32_t i0 = indices[t * 3 + 0];
		const uint32_t i1 = indices[t * 3 + 1];
		const uint32_t i2 = indices[t * 3 + 2];

		const glm::vec3& v0 = positions[i0];
		const glm::vec3& v1 = positions[i1];
		const glm::vec3& v2 = positions[i2];

		const glm::vec3 edge1 = v1 - v0;
		const glm::vec3 edge2 = v2 - v0;
		const float triArea = glm::length(glm::cross(edge1, edge2)) * 0.5f;

		vertexAreas[i0] += triArea / 3.0f;
		vertexAreas[i1] += triArea / 3.0f;
		vertexAreas[i2] += triArea / 3.0f;
	}

	return vertexAreas;
}

inline glm::ivec3 computeGridDimensions(const glm::vec3& gridMin, const glm::vec3& gridMax, float cellSize) {
	const float safeCellSize = std::max(cellSize, 1e-8f);
	const glm::vec3 gridSize = gridMax - gridMin;
	return glm::ivec3(
		std::max(1, static_cast<int>(std::ceil(gridSize.x / safeCellSize))),
		std::max(1, static_cast<int>(std::ceil(gridSize.y / safeCellSize))),
		std::max(1, static_cast<int>(std::ceil(gridSize.z / safeCellSize))));
}

inline bool intersectRayAabb(const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& boundsMin, const glm::vec3& boundsMax,
	float maxDistance, float& outEnter, float& outExit) {
	outEnter = 0.0f;
	outExit = maxDistance;

	for (int axis = 0; axis < 3; ++axis) {
		const float dirAxis = direction[axis];
		if (std::abs(dirAxis) <= 1e-8f) {
			if (origin[axis] < boundsMin[axis] || origin[axis] > boundsMax[axis]) {
				return false;
			}
			continue;
		}

		const float invDir = 1.0f / dirAxis;
		float t0 = (boundsMin[axis] - origin[axis]) * invDir;
		float t1 = (boundsMax[axis] - origin[axis]) * invDir;
		if (t0 > t1) {
			std::swap(t0, t1);
		}

		outEnter = std::max(outEnter, t0);
		outExit = std::min(outExit, t1);
		if (outEnter > outExit) {
			return false;
		}
	}

	return true;
}

inline glm::vec3 barycentricToPosition(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& bary) {
	return (p0 * bary.x) + (p1 * bary.y) + (p2 * bary.z);
}
