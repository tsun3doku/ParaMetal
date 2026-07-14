#include "ContactMapping.hpp"
#include "util/GeometryUtils.hpp"

#include "spatial/TriangleHashGrid.hpp"

#include <cmath>
#include <limits>
#include <algorithm>

ContactMesh buildContactMesh(
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& normals,
    const std::vector<uint32_t>& indices) {
    ContactMesh mesh;
    if (positions.empty() || normals.size() != positions.size() || indices.size() < 3) {
        return mesh;
    }

    mesh.vertices.resize(positions.size());
    for (size_t vertexId = 0; vertexId < positions.size(); ++vertexId) {
        mesh.vertices[vertexId] = {positions[vertexId], normals[vertexId]};
    }
    mesh.indices = indices;
    mesh.triangles.reserve(indices.size() / 3);
    for (size_t index = 0; index + 2 < indices.size(); index += 3) {
        ContactTriangle triangle{};
        triangle.vertexIndices[0] = indices[index];
        triangle.vertexIndices[1] = indices[index + 1];
        triangle.vertexIndices[2] = indices[index + 2];
        if (triangle.vertexIndices[0] < positions.size() &&
            triangle.vertexIndices[1] < positions.size() &&
            triangle.vertexIndices[2] < positions.size()) {
            const glm::vec3 cross = glm::cross(
                positions[triangle.vertexIndices[1]] - positions[triangle.vertexIndices[0]],
                positions[triangle.vertexIndices[2]] - positions[triangle.vertexIndices[0]]);
            const float twiceArea = glm::length(cross);
            triangle.area = 0.5f * twiceArea;
            if (twiceArea > 1e-12f) {
                triangle.normal = cross / twiceArea;
            }
        }
        mesh.triangles.push_back(triangle);
    }
    return mesh;
}

void buildContactPairs(
    const ContactMesh& modelAMesh,
    const std::array<float, 16>& modelALocalToWorld,
    const ContactMesh& modelBMesh,
    const std::array<float, 16>& modelBLocalToWorld,
    std::vector<ContactPair>& contactPairs,
    std::vector<ContactLineVertex>& outOutlineVertices,
    std::vector<ContactLineVertex>& outCorrespondenceVertices,
    float contactRadius,
    float minNormalDot) {
    contactPairs.clear();
    outOutlineVertices.clear();
    outCorrespondenceVertices.clear();

	if (modelAMesh.triangles.empty()) {
		return;
	}

    std::vector<glm::vec3> modelAPositions;
    modelAPositions.reserve(modelAMesh.vertices.size());
    glm::vec3 modelABoundsMin(std::numeric_limits<float>::infinity());
    glm::vec3 modelABoundsMax(-std::numeric_limits<float>::infinity());
    for (const ContactVertex& vertex : modelAMesh.vertices) {
        modelAPositions.push_back(vertex.position);
        modelABoundsMin = glm::min(modelABoundsMin, vertex.position);
        modelABoundsMax = glm::max(modelABoundsMax, vertex.position);
    }

    const float contactPadding = std::max(contactRadius, 1e-4f);
    const glm::vec3 contactPaddingVec(contactPadding);
    TriangleHashGrid modelATriangleGrid;
    modelATriangleGrid.build(
        modelAPositions,
        modelAMesh.indices,
        modelABoundsMin - contactPaddingVec,
        modelABoundsMax + contactPaddingVec,
        contactPadding);

    std::vector<size_t> candidateTriangles;

    glm::mat4 modelAMatrix = toMat4(modelALocalToWorld);
    glm::mat4 inverseModelAMatrix = glm::inverse(modelAMatrix);
	glm::mat3 modelANormalMatrix = glm::transpose(glm::inverse(glm::mat3(modelAMatrix)));

	contactPairs.assign(modelBMesh.triangles.size(), ContactPair{});
	glm::mat4 modelBMatrix = toMat4(modelBLocalToWorld);
		glm::mat3 modelBNormalMatrix = glm::transpose(glm::inverse(glm::mat3(modelBMatrix)));

		for (size_t triIdx = 0; triIdx < modelBMesh.triangles.size(); triIdx++) {
			const auto& modelBTriangle = modelBMesh.triangles[triIdx];
			uint32_t b0 = modelBTriangle.vertexIndices[0];
			uint32_t b1 = modelBTriangle.vertexIndices[1];
			uint32_t b2 = modelBTriangle.vertexIndices[2];
			if (b0 >= modelBMesh.vertices.size() || b1 >= modelBMesh.vertices.size() || b2 >= modelBMesh.vertices.size()) {
				continue;
			}

			const glm::vec3 p0 = modelBMesh.vertices[b0].position;
			const glm::vec3 p1 = modelBMesh.vertices[b1].position;
			const glm::vec3 p2 = modelBMesh.vertices[b2].position;

			const glm::vec3 n0 = modelBMesh.vertices[b0].normal;
			const glm::vec3 n1 = modelBMesh.vertices[b1].normal;
			const glm::vec3 n2 = modelBMesh.vertices[b2].normal;

			ContactPair pair{};
			for (uint32_t si = 0; si < Quadrature::count; ++si) {
				pair.samples[si].modelATriangleIndex = UINT32_MAX;
				pair.samples[si].u = 0.0f;
				pair.samples[si].v = 0.0f;
				pair.samples[si].contactSampleArea = 0.0f;
			}
			pair.contactArea = 0.0f;

			const float modelBTriangleArea = modelBTriangle.area;
			if (modelBTriangleArea <= 0.0f) {
				contactPairs[triIdx] = pair;
				continue;
			}

			for (uint32_t si = 0; si < Quadrature::count; ++si) {
				const glm::vec3 barycentricCoord = Quadrature::bary[si];
				glm::vec3 localPos = p0 * barycentricCoord.x + p1 * barycentricCoord.y + p2 * barycentricCoord.z;
				glm::vec3 localN = n0 * barycentricCoord.x + n1 * barycentricCoord.y + n2 * barycentricCoord.z;
				float localNLen2 = glm::dot(localN, localN);
				if (localNLen2 < 1e-12f) {
					continue;
				}
				localN *= (1.0f / std::sqrt(localNLen2));

				glm::vec3 worldPos = glm::vec3(modelBMatrix * glm::vec4(localPos, 1.0f));
				glm::vec3 worldN = safeNormalize(glm::vec3(modelBNormalMatrix * localN));
				if (glm::dot(worldN, worldN) < 1e-12f) {
					continue;
				}

				glm::vec3 modelAPosition = glm::vec3(inverseModelAMatrix * glm::vec4(worldPos, 1.0f));
				glm::vec3 modelADirection = safeNormalize(glm::vec3(inverseModelAMatrix * glm::vec4(worldN, 0.0f)));
				if (glm::dot(modelADirection, modelADirection) < 1e-12f) {
					continue;
				}

				uint32_t bestTriIdx = UINT32_MAX;
				float bestT = std::numeric_limits<float>::infinity();
				float bestU = 0.0f;
				float bestV = 0.0f;

                candidateTriangles.clear();
                modelATriangleGrid.getTrianglesAlongRay(
                    modelAPosition,
                    modelADirection,
                    contactRadius,
                    candidateTriangles);

				for (size_t candidateTriIdx : candidateTriangles) {
                    const uint32_t modelATriangleIndex = static_cast<uint32_t>(candidateTriIdx);
                    if (modelATriangleIndex >= modelAMesh.triangles.size()) {
                        continue;
                    }
					const auto& modelATriangle = modelAMesh.triangles[modelATriangleIndex];
					uint32_t a0 = modelATriangle.vertexIndices[0];
					uint32_t a1 = modelATriangle.vertexIndices[1];
					uint32_t a2 = modelATriangle.vertexIndices[2];
					if (a0 >= modelAMesh.vertices.size() || a1 >= modelAMesh.vertices.size() || a2 >= modelAMesh.vertices.size()) {
						continue;
					}

					glm::vec3 modelANormalWorld = safeNormalize(glm::vec3(modelANormalMatrix * modelATriangle.normal));
					float nd = glm::dot(worldN, modelANormalWorld);
					if (minNormalDot < 0.0f) {
						if (nd > minNormalDot) {
							continue;
						}
					} else {
						if (nd < minNormalDot) {
							continue;
						}
					}

					const glm::vec3& v0 = modelAMesh.vertices[a0].position;
					const glm::vec3& v1 = modelAMesh.vertices[a1].position;
					const glm::vec3& v2 = modelAMesh.vertices[a2].position;

					float t = 0.0f;
					float u = 0.0f;
					float v = 0.0f;
					if (!intersectRayTriangle(modelAPosition, modelADirection, v0, v1, v2, t, u, v)) {
						continue;
					}
					if (t > contactRadius) {
						continue;
					}
					if (t < bestT) {
						bestT = t;
						bestTriIdx = modelATriangleIndex;
						bestU = u;
						bestV = v;
					}
				}

				if (bestTriIdx != UINT32_MAX) {
					pair.samples[si].modelATriangleIndex = bestTriIdx;
					pair.samples[si].u = bestU;
					pair.samples[si].v = bestV;
					const float contactSampleArea = Quadrature::weights[si] * modelBTriangleArea;
					pair.samples[si].contactSampleArea = contactSampleArea;
					pair.contactArea += contactSampleArea;
				}
			}

			contactPairs[triIdx] = pair;
		}

	outOutlineVertices.reserve(256);
	outCorrespondenceVertices.reserve(256);
		for (size_t triIdx = 0; triIdx < modelBMesh.triangles.size(); triIdx++) {
			if (triIdx >= contactPairs.size()) {
				continue;
			}
			const ContactPair& pair = contactPairs[triIdx];
			if (pair.contactArea <= 0.0f) {
				continue;
			}

			const auto& modelBTriangle = modelBMesh.triangles[triIdx];
			uint32_t b0 = modelBTriangle.vertexIndices[0];
			uint32_t b1 = modelBTriangle.vertexIndices[1];
			uint32_t b2 = modelBTriangle.vertexIndices[2];
			if (b0 >= modelBMesh.vertices.size() || b1 >= modelBMesh.vertices.size() || b2 >= modelBMesh.vertices.size()) {
				continue;
			}

			glm::vec3 modelBWorld0 = glm::vec3(modelBMatrix * glm::vec4(modelBMesh.vertices[b0].position, 1.0f));
			glm::vec3 modelBWorld1 = glm::vec3(modelBMatrix * glm::vec4(modelBMesh.vertices[b1].position, 1.0f));
			glm::vec3 modelBWorld2 = glm::vec3(modelBMatrix * glm::vec4(modelBMesh.vertices[b2].position, 1.0f));

			float contactRatio = 0.0f;
			if (modelBTriangle.area > 0.0f) {
				contactRatio = pair.contactArea / modelBTriangle.area;
			}

			glm::vec3 patchColor = glm::vec3(0.0f, 0.304f, 0.918f); 
			if (contactRatio > 0.95f) {
				patchColor = glm::vec3(1.0f, 0.2f, 0.1f);
			}
			ContactLineVertex l0{};
			l0.position = modelBWorld0;
			l0.color = patchColor;
			ContactLineVertex l1{};
			l1.position = modelBWorld1;
			l1.color = patchColor;
			outOutlineVertices.push_back(l0);
			outOutlineVertices.push_back(l1);

			ContactLineVertex l2{};
			l2.position = modelBWorld1;
			l2.color = patchColor;
			ContactLineVertex l3{};
			l3.position = modelBWorld2;
			l3.color = patchColor;
			outOutlineVertices.push_back(l2);
			outOutlineVertices.push_back(l3);

			ContactLineVertex l4{};
			l4.position = modelBWorld2;
			l4.color = patchColor;
			ContactLineVertex l5{};
			l5.position = modelBWorld0;
			l5.color = patchColor;
			outOutlineVertices.push_back(l4);
			outOutlineVertices.push_back(l5);

			glm::vec3 triN = glm::cross(modelBWorld1 - modelBWorld0, modelBWorld2 - modelBWorld0);
			float triNLen2 = glm::dot(triN, triN);
			if (triNLen2 > 1e-12f) {
				triN *= (1.0f / std::sqrt(triNLen2));
			}
			glm::vec3 uDir = modelBWorld1 - modelBWorld0;
			float uLen2 = glm::dot(uDir, uDir);
			if (uLen2 > 1e-12f) {
				uDir *= (1.0f / std::sqrt(uLen2));
			}
			glm::vec3 vDir = glm::cross(triN, uDir);
			float vLen2 = glm::dot(vDir, vDir);
			if (vLen2 > 1e-12f) {
				vDir *= (1.0f / std::sqrt(vLen2));
			}
			float crossLen = std::sqrt(std::max(1e-12f, modelBTriangle.area)) * 0.1f;

			for (uint32_t si = 0; si < Quadrature::count; ++si) {
				const contact::Sample& s = pair.samples[si];
				if (s.modelATriangleIndex == UINT32_MAX || s.contactSampleArea <= 0.0f) {
					continue;
				}
				if (s.modelATriangleIndex >= modelAMesh.triangles.size()) {
					continue;
				}

				const glm::vec3 barycentricCoord = Quadrature::bary[si];
				glm::vec3 modelBSamplePosition = modelBWorld0 * barycentricCoord.x + modelBWorld1 * barycentricCoord.y + modelBWorld2 * barycentricCoord.z;

				const auto& modelATriangle = modelAMesh.triangles[s.modelATriangleIndex];
				uint32_t a0 = modelATriangle.vertexIndices[0];
				uint32_t a1 = modelATriangle.vertexIndices[1];
				uint32_t a2 = modelATriangle.vertexIndices[2];
				if (a0 >= modelAMesh.vertices.size() || a1 >= modelAMesh.vertices.size() || a2 >= modelAMesh.vertices.size()) {
					continue;
				}

				glm::vec3 modelAWorld0 = glm::vec3(modelAMatrix * glm::vec4(modelAMesh.vertices[a0].position, 1.0f));
				glm::vec3 modelAWorld1 = glm::vec3(modelAMatrix * glm::vec4(modelAMesh.vertices[a1].position, 1.0f));
				glm::vec3 modelAWorld2 = glm::vec3(modelAMatrix * glm::vec4(modelAMesh.vertices[a2].position, 1.0f));
				glm::vec3 modelABarycentric = glm::vec3(1.0f - s.u - s.v, s.u, s.v);
				glm::vec3 modelAHitPosition = modelAWorld0 * modelABarycentric.x +
					modelAWorld1 * modelABarycentric.y + modelAWorld2 * modelABarycentric.z;

				ContactLineVertex a{};
				a.position = modelBSamplePosition;
				a.color = glm::vec3(0.9f, 0.9f, 0.2f);
				ContactLineVertex modelAHitVertex{};
				modelAHitVertex.position = modelAHitPosition;
				modelAHitVertex.color = glm::vec3(1.0f, 0.1f, 0.2f);
				outCorrespondenceVertices.push_back(a);
				outCorrespondenceVertices.push_back(modelAHitVertex);

				ContactLineVertex c0{};
				c0.position = modelBSamplePosition - uDir * crossLen;
				c0.color = glm::vec3(0.9f, 0.9f, 0.2f);
				ContactLineVertex c1{};
				c1.position = modelBSamplePosition + uDir * crossLen;
				c1.color = glm::vec3(0.9f, 0.9f, 0.2f);
				outCorrespondenceVertices.push_back(c0);
				outCorrespondenceVertices.push_back(c1);

				ContactLineVertex d0{};
				d0.position = modelBSamplePosition - vDir * crossLen;
				d0.color = glm::vec3(0.9f, 0.9f, 0.2f);
				ContactLineVertex d1{};
				d1.position = modelBSamplePosition + vDir * crossLen;
				d1.color = glm::vec3(0.9f, 0.9f, 0.2f);
				outCorrespondenceVertices.push_back(d0);
				outCorrespondenceVertices.push_back(d1);
			}
		}
}
